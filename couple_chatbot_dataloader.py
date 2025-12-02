import re  # 正则表达式，用于解析每一行
import random  # 划分训练 / 验证 / 测试集时打乱索引
import unicodedata  # 判断字符类别，用于清洗文本
from datetime import datetime  # 解析时间戳
import torch  # PyTorch 主库
from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器


# ====================== 词表类：不依赖 d2l ======================
class Vocab:
    """自定义词表类，实现 token ↔ id 映射"""

    def __init__(self, tokens, min_freq=1, reserved_tokens=None):
        # tokens: 二维列表，例如 [['早', '呀', '！'], ['对', '的']]
        if reserved_tokens is None:
            reserved_tokens = []

        # 将二维 token 列表展平成一维列表
        tokens = [tk for line in tokens for tk in line]

        # 统计每个 token 的频次
        counter = {}
        for tk in tokens:
            counter[tk] = counter.get(tk, 0) + 1

        # 按频次从高到低排序
        sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 先放入预留的特殊 token
        self.idx_to_token = list(reserved_tokens)  # id → token
        self.token_to_idx = {tk: i for i, tk in enumerate(self.idx_to_token)}  # token → id

        # 再放入普通 token（频次小于 min_freq 的跳过）
        for tk, freq in sorted_tokens:
            if freq < min_freq:
                continue
            if tk in self.token_to_idx:
                continue
            self.token_to_idx[tk] = len(self.idx_to_token)
            self.idx_to_token.append(tk)

        # 确保一定有 <unk>，用于处理未登录词
        if '<unk>' not in self.token_to_idx:
            self.token_to_idx['<unk>'] = len(self.idx_to_token)
            self.idx_to_token.append('<unk>')

    def __len__(self):
        """返回词表大小"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """支持 vocab[token] 或 vocab[token_list]"""
        if isinstance(tokens, list):
            return [self.__getitem__(tk) for tk in tokens]
        return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])

    def to_tokens(self, indices):
        """id 或 id 列表 → token 或 token 列表"""
        if isinstance(indices, list):
            return [self.to_tokens(i) for i in indices]
        if 0 <= indices < len(self.idx_to_token):
            return self.idx_to_token[indices]
        return '<unk>'


# ====================== 截断 / 填充函数 ======================
def truncate_pad(line, num_steps, padding_id):
    """将序列截断或填充到固定长度 num_steps"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_id] * (num_steps - len(line))


# ====================== 文本清洗：去掉控制字符和奇怪符号 ======================
def clean_text(text):
    """清理掉控制字符、占位符（表情/图片/链接）以及奇怪字符"""

    # ① 先干掉像 "图片]"、"捂脸]"、"月亮]" 这种残留占位符
    #    匹配：连续若干非空白字符 + 右中括号，例如 "图片]"、"捂脸]"、"xxx]"
    text = re.sub(r"[^\s\]]+]", "", text)

    # ② 再过滤微信导出的完整占位符
    text = re.sub(r"\[表情\]", "", text)
    text = re.sub(r"\[链接\]", "", text)
    # 如果你想一刀切所有 [xxx]，可用下面这行替换上面三行：
    # text = re.sub(r"\[[^\]]+\]", "", text)

    # ③ 去掉控制字符、未定义字符
    cleaned_chars = []
    for ch in text:
        if unicodedata.category(ch).startswith('C'):
            continue
        cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # ④ 只保留常见中文、英文、数字、标点
    text = re.sub(
        r"[^\u4e00-\u9fffA-Za-z0-9，。、“”\"'！？!？…—：:；;（）()\[\]\s]",
        "",
        text
    )

    # ⑤ 压缩连续空白符
    text = re.sub(r"\s+", " ", text).strip()

    # ⑥ 去掉句首一长串“不是中文 / 字母 / 数字”的前缀（清理 (!!M 这类垃圾）
    text = re.sub(r"^[^\u4e00-\u9fffA-Za-z0-9]+", "", text)

    return text



# ====================== 读取 .dat 聊天记录（包含时间戳） ======================
def read_chat_records(file_path):
    """
    读取微信 .dat 聊天记录，解析为 (speaker, text, timestamp) 列表
    timestamp 为 datetime 对象，便于按时间切分 session
    """
    # 先尝试几种常见编码：utf-8 / utf-8-sig / gb18030
    encodings_to_try = ["utf-8", "utf-8-sig", "gb18030"]
    lines = None

    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                lines = f.readlines()
            print(f"[Info] 使用编码 {enc} 成功读取文件")
            break
        except UnicodeDecodeError as e:
            print(f"[Warn] 使用编码 {enc} 读取失败：{e}")

    # 如果上述编码都失败，则使用 utf-8 + ignore 强行读取
    if lines is None:
        print("[Warn] 尝试多种编码均失败，改用 utf-8(errors='ignore') 读取，"
              "个别乱码字符会被丢弃")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

    records = []  # 用于保存 (speaker, text, datetime) 的列表

    # 典型行格式：
    # Uxxx (2019-10-22 23:24:09):对了 ，我发现你歌单里的这首歌特好听
    # Txxx (2019-10-22 23:27:40):也好暖
    pattern = re.compile(
        r"(.+?) \((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\):(.*)"
    )

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        m = pattern.match(line)
        if not m:
            # 不匹配正常聊天格式的行（如系统消息）忽略
            continue

        raw_name = m.group(1).strip()         # 带前缀的名字
        time_str = m.group(2).strip()         # 时间字符串
        content = m.group(3).strip()          # 消息内容

        # 解析时间戳
        try:
            timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # 时间解析失败的行直接跳过
            continue

        # 去掉名字前面的前缀字符（U/T/`/6 等）
        if len(raw_name) >= 2:
            speaker = raw_name[1:].strip()
        else:
            speaker = raw_name

        # 文本清洗
        content = clean_text(content)
        if not content:
            continue

        records.append((speaker, content, timestamp))

    # 按时间排序保证稳妥（一般文件本身已经按时间排好，这里是保险）
    records.sort(key=lambda x: x[2])

    return records


# ====================== 构造 (src, tgt) 句对：带时间 + 状态机 ======================
def build_src_tgt_pairs(
    records,
    src_keyword,           # 编码器输入方昵称关键字
    tgt_keyword,           # 解码器输出方昵称关键字
    session_gap_seconds=3600   # 会话切分阈值：1 小时 = 3600 秒
):

    pairs = []                    # 存放 (src, tgt) 句对
    pending_src = None            # 当前轮中，累计发言
    pending_tgt = None            # 当前轮中，累计回复
    state = "idle"                # 状态：idle / collecting_src / collecting_tgt
    last_time = None              # 上一条消息的时间

    for speaker, text, ts in records:
        # 1. 时间切分：如果距离上一条消息 ≥ session_gap_seconds，则视为新 session
        if last_time is not None:
            gap = (ts - last_time).total_seconds()
            if gap >= session_gap_seconds:
                # 先把上一轮（如果完整）收尾
                if pending_src and pending_tgt:
                    pairs.append((pending_src.strip(), pending_tgt.strip()))
                # 然后重置整个状态机
                pending_src, pending_tgt = None, None
                state = "idle"
        last_time = ts  # 更新上一条消息时间

        # 2. 根据说话人和当前状态更新状态机
        # ========= 发言 =========
        if src_keyword in speaker:
            if state == "idle":
                pending_src = text
                pending_tgt = None
                state = "collecting_src"

            elif state == "collecting_src":
                pending_src = (pending_src + " " + text).strip()

            elif state == "collecting_tgt":
                if pending_src and pending_tgt:
                    pairs.append((pending_src.strip(), pending_tgt.strip()))
                pending_src = text
                pending_tgt = None
                state = "collecting_src"

        # ========= 发言 =========
        elif tgt_keyword in speaker:
            if state == "idle":
                continue

            elif state == "collecting_src":
                pending_tgt = text
                state = "collecting_tgt"

            elif state == "collecting_tgt":
                pending_tgt = (pending_tgt + " " + text).strip()

        # ========= 其他人 或 其他情况 =========
        else:
            # 如果聊天记录里有第三人，这里直接忽略
            continue

    # 遍历结束后，可能还有一轮未写入的 pair
    if pending_src and pending_tgt:
        pairs.append((pending_src.strip(), pending_tgt.strip()))

    return pairs


# ====================== 文本 → 字符级 token 序列 ======================
def char_tokenize_pairs(pairs):
    """将句对列表转换为字符级 token 序列列表"""
    src_lines = []  # 源句子的 token 序列列表
    tgt_lines = []  # 目标句子的 token 序列列表

    for src_text, tgt_text in pairs:
        src_tokens = list(src_text)  # 按字符拆分源句子
        tgt_tokens = list(tgt_text)  # 按字符拆分目标句子
        src_lines.append(src_tokens)
        tgt_lines.append(tgt_tokens)

    return src_lines, tgt_lines


# ====================== 将 token 序列转换为张量 + 有效长度 ======================
def build_array_seq2seq(lines, vocab, num_steps):
    """将 token 序列列表转换成张量和有效长度"""
    # 先将 token 序列转换为 id 序列，并在末尾加 <eos>
    ids = [vocab[line] + [vocab['<eos>']] for line in lines]

    # 对 id 序列进行截断 / 填充
    array = torch.tensor([
        truncate_pad(seq, num_steps, vocab['<pad>'])
        for seq in ids
    ], dtype=torch.long)

    # 有效长度：非 pad 的元素个数
    valid_len = (array != vocab['<pad>']).sum(dim=1)

    return array, valid_len


# ====================== 构造 DataLoader ======================
def make_loader(src_array, src_len, tgt_array, tgt_len,
                batch_size, shuffle):
    """根据四个张量构建 DataLoader"""
    dataset = TensorDataset(src_array, src_len, tgt_array, tgt_len)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=0)
    return loader


# ====================== 对外主接口：返回 dataloader + vocab ======================
def get_seq2seq_dataloaders(
    file_path,          # .dat 文件路径
    batch_size=64,      # batch 大小
    num_steps=50,       # 每个序列最大长度
    train_ratio=0.8,    # 训练集比例
    val_ratio=0.1,      # 验证集比例（测试集 = 剩余）
    src_min_freq=1,     # 源 vocab 最小词频
    tgt_min_freq=1,     # 目标 vocab 最小词频
    src_keyword=None,   # 源方昵称关键字（必填）
    tgt_keyword=None    # 目标方昵称关键字（必填）
):
    """从微信 .dat 文件构造 train/val/test DataLoader 和词表"""

    # ---------- 1. 读取原始聊天记录（含时间戳） ----------
    records = read_chat_records(file_path)
    print(f"[Info] 解析到聊天记录条数：{len(records)}")

    if src_keyword is None or tgt_keyword is None:
        raise ValueError("必须显式传入 src_keyword 和 tgt_keyword（微信昵称）")

    # ---------- 2. 构造 (src, tgt) 句对（带时间 + 状态机） ----------
    pairs = build_src_tgt_pairs(records,
                                src_keyword=src_keyword,
                                tgt_keyword=tgt_keyword)
    print(f"[Info] 构造得到句对数量：{len(pairs)}")
    if len(pairs) == 0:
        print("[Warning] 句对数量为 0，请检查关键字或原始数据格式。")

    # ---------- 3. 字符级 token 化 ----------
    src_lines, tgt_lines = char_tokenize_pairs(pairs)

    # ---------- 4. 构建 vocab ----------
    reserved = ['<pad>', '<bos>', '<eos>', '<unk>']
    src_vocab = Vocab(src_lines, min_freq=src_min_freq, reserved_tokens=reserved)
    tgt_vocab = Vocab(tgt_lines, min_freq=tgt_min_freq, reserved_tokens=reserved)
    print(f"[Info] 源 vocab 大小：{len(src_vocab)}，目标 vocab 大小：{len(tgt_vocab)}")

    # ---------- 5. 将 token 序列转换为张量 ----------
    src_array, src_valid_len = build_array_seq2seq(src_lines, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_seq2seq(tgt_lines, tgt_vocab, num_steps)

    num_samples = src_array.shape[0]
    indices = list(range(num_samples))
    random.shuffle(indices)

    # ---------- 6. 划分训练 / 验证 / 测试 ----------
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def _select(idx_list):
        idx_tensor = torch.tensor(idx_list, dtype=torch.long)
        return (
            src_array[idx_tensor],
            src_valid_len[idx_tensor],
            tgt_array[idx_tensor],
            tgt_valid_len[idx_tensor],
        )

    train_src, train_src_len, train_tgt, train_tgt_len = _select(train_idx)
    val_src, val_src_len, val_tgt, val_tgt_len = _select(val_idx)
    test_src, test_src_len, test_tgt, test_tgt_len = _select(test_idx)

    # ---------- 7. 构建 DataLoader ----------
    train_loader = make_loader(train_src, train_src_len, train_tgt, train_tgt_len,
                               batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_src, val_src_len, val_tgt, val_tgt_len,
                             batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_src, test_src_len, test_tgt, test_tgt_len,
                              batch_size=batch_size, shuffle=False)

    print(f"[Info] train / val / test 样本数：{len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


# ====================== 小工具：将 id 序列转回句子（方便调试） ======================
def ids_to_sentence(id_tensor, vocab):
    """将一维 id 张量转换为可读文本句子"""
    tokens = []
    for idx in id_tensor.tolist():
        token = vocab.to_tokens(idx)
        if token == '<pad>':
            continue
        if token == '<eos>':
            break
        tokens.append(token)
    # 字符级：直接拼接
    return "".join(tokens)


# ====================== 直接运行本文件时的简单自测 ======================
if __name__ == "__main__":
    # 你可以在这里修改 .dat 文件路径
    data_file = "./wxid_osb3shozax2v22.dat"  # 请确认路径正确

    # 在自测模式下交互式输入双方昵称
    input_name = input("【自测】请输入你想要扮演的一方的微信昵称：")
    output_name = input("【自测】请输入对方的微信昵称：")

    # 调用主接口构建 dataloader 和 vocab
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_seq2seq_dataloaders(
        file_path=data_file,
        batch_size=10,
        num_steps=50,
        src_keyword=input_name,
        tgt_keyword=output_name
    )

    # 打印一个 batch 看看效果
    print("\n[Debug] 打印第一个 batch 的原始张量：")
    for batch in train_loader:
        src, src_len, tgt, tgt_len = batch
        print("src:\n", src)
        print("src_len:\n", src_len)
        print("tgt:\n", tgt)
        print("tgt_len:\n", tgt_len)
        break

    # 将第一个 batch 转回可读文本，检查 src → tgt 是否合理
    print("\n[Debug] 将第一个 batch 转回可读文本：")
    for i in range(src.shape[0]):
        src_sentence = ids_to_sentence(src[i], src_vocab)
        tgt_sentence = ids_to_sentence(tgt[i], tgt_vocab)
        print(f"[样本 {i}]")
        print("SRC:", src_sentence)
        print("TGT:", tgt_sentence)
        print("-" * 40)