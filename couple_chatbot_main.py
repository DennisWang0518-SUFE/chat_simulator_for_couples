import math  # 导入数学库（用于BLEU的指数等运算）
import collections  # 导入collections库（BLEU中统计n-gram用）
import torch  # 导入PyTorch主库
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard日志记录器
from torch.optim.lr_scheduler import CosineAnnealingLR  # 导入余弦退火学习率调度器
from couple_chatbot_seq2seq import Seq2SeqEncoder, Seq2SeqAttentionDecoder, Seq2Seq  # 导入自定义的编码器、解码器和整体Seq2Seq模型
from couple_chatbot_dataloader import get_seq2seq_dataloaders  # 放在文件开头

# 获取聊天记录文件
date_file_name = input("当前目录下聊天记录（.dat文件）的文件名是：")
# 获取输入方和输出方昵称
input_name = input("请输入你想要扮演的一方的微信昵称：")
output_name = input("请输入对方的微信昵称：")


# ====================== 工具函数：梯度裁剪 ======================
def grad_clipping(net, max_norm):  # 定义梯度裁剪函数，防止梯度爆炸
    nn.utils.clip_grad_norm_(net.parameters(), max_norm)  # 使用PyTorch内置的clip_grad_norm_按范数裁剪梯度


# ====================== 工具函数：序列mask ======================
def sequence_mask(X, valid_len, value=0):  # 定义序列mask函数，用于在loss中屏蔽padding部分
    maxlen = X.size(1)  # 获取序列的最大长度（时间步数）
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_len[:, None]  # 生成形状为(batch, maxlen)的布尔mask
    X = X.clone()  # 克隆一份X，避免就地修改上游计算图
    X[~mask] = value  # 对mask为False的位置赋值为指定的value（例如0）
    return X  # 返回加了mask后的张量


# ====================== 损失函数：带mask的交叉熵 ======================
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):  # 定义带mask的交叉熵损失类，继承自nn.CrossEntropyLoss
    # pred: (batch_size, num_steps, vocab_size)
    # label: (batch_size, num_steps)
    # valid_len: (batch_size,)
    def forward(self, pred, label, valid_len):  # 重写forward实现
        weights = torch.ones_like(label, dtype=torch.float32)  # 初始化权重矩阵，全1，形状与label相同
        weights = sequence_mask(weights, valid_len)  # 对权重矩阵按有效长度进行mask，padding位置为0
        self.reduction = 'none'  # 禁用父类默认的'reduction'，保留逐时间步的loss
        # CrossEntropyLoss期望输入形状为(batch, vocab_size, num_steps)，所以需要permute
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label
        )  # 计算未加权loss，shape为(batch_size, num_steps)
        weighted_loss = (unweighted_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)  # 对有效时间步加权求平均
        return weighted_loss  # 返回每个样本的平均loss，shape为(batch_size,)


# ====================== BLEU 计算函数 ======================
def bleu(pred_seq, label_seq, k):  # 定义BLEU函数，k表示考虑1~k阶n-gram
    pred_tokens = pred_seq.split(' ')  # 将预测句按空格切分成token列表
    label_tokens = label_seq.split(' ')  # 将标签句按空格切分成token列表
    len_pred, len_label = len(pred_tokens), len(label_tokens)  # 获取预测句和标签句的长度
    # 短句惩罚因子（brevity penalty）
    score = math.exp(min(0, 1 - len_label / max(len_pred, 1)))  # 若预测更短则会有惩罚，避免除零用max
    # 从1-gram到k-gram依次计算精度
    for n in range(1, k + 1):  # 遍历n-gram阶数
        num_matches = 0  # 匹配到的n-gram数量
        label_subs = collections.defaultdict(int)  # 统计标签序列中各n-gram出现次数的字典
        # 统计标签中的所有n-gram
        for i in range(len_label - n + 1):  # 遍历所有可能起点
            ngram = ' '.join(label_tokens[i: i + n])  # 取出长度为n的n-gram
            label_subs[ngram] += 1  # 该n-gram计数+1
        # 遍历预测序列中的n-gram，统计匹配数
        for i in range(len_pred - n + 1):  # 同样遍历预测n-gram
            ngram = ' '.join(pred_tokens[i: i + n])  # 预测中的n-gram
            if label_subs[ngram] > 0:  # 若在标签统计中还有剩余次数
                num_matches += 1  # 匹配计数+1
                label_subs[ngram] -= 1  # 标签中的该n-gram计数减1，避免重复匹配
        denom = max(len_pred - n + 1, 1)  # 预测中n-gram总数，防止除零
        precision = num_matches / denom  # n-gram精度
        score *= math.pow(precision, 0.5 ** n)  # BLEU使用加权几何平均，这里权重设为1/2^n
    return score  # 返回最终BLEU得分


# ====================== 工具函数：id序列转句子 ======================
def ids_to_sentence(id_seq, vocab):  # 将id序列转换为文本句子（以空格拼接）
    # 这里假设vocab有 idx_to_token 列表属性：idx_to_token[id] = token
    tokens = []  # 初始化token列表
    for idx in id_seq:  # 遍历序列中的每个id
        token = vocab.idx_to_token[int(idx)]  # 通过idx_to_token从id映射到token
        if token == '<bos>':  # 跳过起始标记
            continue  # 继续下一个id
        if token == '<eos>':  # 遇到结束标记则停止
            break  # 结束循环
        if token == '<pad>':  # 跳过padding标记
            continue  # 继续下一个id
        tokens.append(token)  # 将有效token加入列表
    return ' '.join(tokens)  # 用空格拼接为一句话


# ====================== 单句预测函数 ======================
def translate_sentence(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):  # 定义单句翻译函数
    net.eval()  # 将模型切换到评估模式（关闭dropout等）
    # 将输入句子按空格切分并小写，这里假设src_vocab[token]可以返回id
    src_tokens = [src_vocab[token] for token in src_sentence.lower().split(' ')]  # 将源句token映射为id列表
    src_tokens.append(src_vocab['<eos>'])  # 在末尾添加结束标记<eos>
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)  # 构造源句有效长度张量
    # 对源句进行pad或截断到固定num_steps长度，这里用简单方式手写truncate/pad
    if len(src_tokens) > num_steps:  # 若长度超过最大步数
        src_tokens = src_tokens[:num_steps]  # 截断到num_steps
    else:  # 若长度不足
        src_tokens = src_tokens + [src_vocab['<pad>']] * (num_steps - len(src_tokens))  # 用<pad>补齐到num_steps
    enc_X = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)  # 转为张量并增加batch维度(1, num_steps)
    # 前向编码得到encoder输出
    enc_outputs = net.encoder(enc_X, enc_valid_len)  # 调用encoder得到输出和最终隐状态
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)  # 用encoder的隐状态初始化decoder状态
    # 解码器初始输入为<bos>标记
    dec_X = torch.tensor([[tgt_vocab['<bos>']]], dtype=torch.long, device=device)  # 形状(1,1)
    output_ids = []  # 用于保存生成的目标序列id
    for _ in range(num_steps):  # 逐时间步生成最多num_steps个token
        Y, dec_state = net.decoder(dec_X, dec_state)  # 调用decoder得到当前步输出和新的状态
        dec_X = Y.argmax(dim=2)  # 对输出在vocab维度上取argmax作为下一个输入id，形状仍为(1,1)
        pred_id = int(dec_X.squeeze(0).squeeze(0).item())  # 取出具体的整数id
        if pred_id == tgt_vocab['<eos>']:  # 若生成<eos>则停止
            break  # 结束生成
        output_ids.append(pred_id)  # 将生成的id加入列表
    translation = ids_to_sentence(output_ids, tgt_vocab)  # 将id序列转换为可读文本
    return translation  # 返回翻译结果字符串


# ====================== 在验证/测试集上评估BLEU ======================
def evaluate_bleu_on_loader(net, data_loader, src_vocab, tgt_vocab, num_steps, device, k=2):  # 定义在dataloader上统计BLEU的函数
    net.eval()  # 切换到eval模式
    total_bleu = 0.0  # 累积BLEU得分
    count = 0  # 样本计数
    with torch.no_grad():  # 评估阶段关闭梯度计算
        for batch in data_loader:  # 遍历数据加载器中的所有batch
            # 这里假设batch = (src, src_len, tgt, tgt_len)
            src, src_len, tgt, tgt_len = batch  # 解包batch中四个张量
            src = src.to(device)  # 将源序列移到device
            src_len = src_len.to(device)  # 将源有效长度移到device
            tgt = tgt.to(device)  # 将目标序列移到device
            tgt_len = tgt_len.to(device)  # 将目标有效长度移到device
            batch_size = src.shape[0]  # 获取当前batch大小
            # 对batch中每个样本单独生成翻译并计算BLEU（模板写法，真实使用可优化）
            for i in range(batch_size):  # 遍历batch中每个样本
                # 取单条源句id序列（长度为num_steps）
                src_ids = src[i].tolist()  # 将第i条源序列转为Python列表
                # 根据有效长度裁剪有效部分
                valid_l = int(src_len[i].item())  # 取出该样本的有效长度
                src_ids_trim = src_ids[:valid_l]  # 根据有效长度截断
                # 将id序列映射回token再拼接成句子
                src_sentence = ids_to_sentence(src_ids_trim, src_vocab)  # 获取可读的源句文本
                # 使用模型进行翻译
                pred_sentence = translate_sentence(net, src_sentence, src_vocab, tgt_vocab, num_steps, device)  # 获取预测句
                # 处理标签句：将id序列转换为文本
                tgt_ids = tgt[i].tolist()  # 获取第i条目标id序列
                tgt_sentence = ids_to_sentence(tgt_ids, tgt_vocab)  # 将目标id转为文本句子
                # 计算该句子的BLEU
                score = bleu(pred_sentence, tgt_sentence, k=k)  # 计算BLEU得分
                total_bleu += score  # 累加BLEU
                count += 1  # 样本数+1
    avg_bleu = total_bleu / max(count, 1)  # 计算平均BLEU，防止除零
    return avg_bleu  # 返回平均BLEU


# ====================== 训练函数 ======================
def train_seq2seq():  # 定义主训练函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU作为训练设备

    # ======== 1. 创建日志工具 ========
    writer = SummaryWriter("logs_seq2seq_train")  # 创建TensorBoard日志记录器，日志保存到指定目录

    # ======== 2. 准备数据：交给你自己实现 ========
    # 你需要自己写一个函数 get_seq2seq_dataloaders 返回：
    # train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    # 这里示例写法：请在实际项目中用你自己的导入代码替换
    # train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_seq2seq_dataloaders()
    # raise NotImplementedError("请实现并导入 get_seq2seq_dataloaders，然后注释掉这一行")  # 提示你自行实现dataloader
    data_file = f"./{date_file_name}.dat"  # 你的 dat 文件路径
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_seq2seq_dataloaders(
        file_path=data_file,
        batch_size=64,
        num_steps=50,
        src_keyword=input_name,
        tgt_keyword=output_name
    )

    torch.save({
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "input_name": input_name,
        "output_name": output_name
    }, "vocab_dict.pth")

    print(f"[Info] 词表已保存，同时保存昵称：input={input_name}, output={output_name}")

    # ======== 3. 超参数设置 ========
    embed_size = 256  # 词向量维度
    num_hiddens = 512  # RNN隐藏状态维度
    num_layers = 2  # RNN层数
    dropout = 0.1  # RNN中的dropout比例
    lr = 1e-3  # 初始学习率
    num_epochs = 10  # 训练轮数
    grad_clip = 1.0  # 梯度裁剪阈值
    eval_every = 1  # 每多少个epoch在验证集上评估一次

    # ======== 4. 构建模型 ========
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = Seq2Seq(encoder, decoder).to(device)  # 将编码器和解码器封装为整体Seq2Seq模型并移到device

    # ======== 5. 创建损失函数与优化器、学习率调度器 ========
    loss_fn = MaskedSoftmaxCELoss()  # 使用带mask的交叉熵损失
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 使用Adam作为优化器
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)  # 使用余弦退火调整学习率

    # ======== 6. 一些训练过程监控变量 ========
    global_step = 0  # 全局训练步数计数器
    best_val_loss = float("inf")  # 记录验证集上最小的loss
    best_ckpt_path = "best_seq2seq_checkpoint.pth"  # 模型参数checkpoint保存路径
    best_model_path = "best_seq2seq_model.pth"  # 完整模型保存路径

    # ======== 7. 开始epoch级训练循环 ========
    for epoch in range(num_epochs):  # 遍历每一个epoch
        net.train()  # 切换到训练模式
        epoch_loss_sum = 0.0  # 当前epoch损失累加
        epoch_token_count = 0  # 当前epoch有效token数量累加

        for batch in train_loader:  # 遍历训练集中每个batch
            global_step += 1  # 全局步数+1
            # 这里假设batch = (src, src_len, tgt, tgt_len)
            src, src_len, tgt, tgt_len = batch  # 解包batch
            src = src.to(device)  # 将源序列移到device
            src_len = src_len.to(device)  # 将源有效长度移到device
            tgt = tgt.to(device)  # 将目标序列移到device
            tgt_len = tgt_len.to(device)  # 将目标有效长度移到device

            # 构造decoder输入：在目标序列左侧添加<bos>，并去掉最后一个token（即右移一位）
            bos_ids = torch.full((tgt.shape[0], 1), tgt_vocab['<bos>'], dtype=torch.long, device=device)  # 构建<bos>列
            dec_input = torch.cat([bos_ids, tgt[:, :-1]], dim=1)  # 将<bos>拼接到tgt左侧，得到decoder输入

            optimizer.zero_grad()  # 梯度清零
            # 前向传播：传入源序列、decoder输入以及源序列有效长度
            pred = net(src, dec_input, src_len)  # 得到预测序列logits，形状为(batch, num_steps, vocab_size)
            # 计算损失：传入预测logits、真实目标序列和目标有效长度
            loss = loss_fn(pred, tgt, tgt_len)  # 得到每个样本的loss，形状为(batch,)
            loss_sum = loss.sum()  # 将batch内样本loss做求和
            loss_sum.backward()  # 对总loss反向传播，计算梯度
            grad_clipping(net, grad_clip)  # 对模型参数的梯度进行裁剪
            optimizer.step()  # 更新模型参数

            # 统计当前epoch的损失和token数
            epoch_loss_sum += loss_sum.item()  # 累加本batch总loss
            epoch_token_count += int(tgt_len.sum().item())  # 累加本batch有效token总数

            # 定期打印训练日志
            if global_step % 50 == 0:  # 每训练50个step打印一次
                avg_loss = epoch_loss_sum / max(epoch_token_count, 1)  # 计算当前epoch的平均loss/token
                print(f"[Epoch {epoch + 1}/{num_epochs}] Step {global_step}, "
                      f"Train loss/token: {avg_loss:.4f}")  # 打印训练信息
                writer.add_scalar("Train/loss_per_token", avg_loss, global_step)  # 将训练loss写入TensorBoard

        # 每个epoch结束后更新学习率调度器
        scheduler.step()  # 调用学习率调度器进行一步更新

        # ======== 8. 在验证集上评估 ========
        if (epoch + 1) % eval_every == 0:  # 每eval_every个epoch做一次验证
            net.eval()  # 切换到评估模式
            val_loss_sum = 0.0  # 验证集loss累加
            val_token_count = 0  # 验证集token数累加
            with torch.no_grad():  # 关闭梯度
                for batch in val_loader:  # 遍历验证集
                    src, src_len, tgt, tgt_len = batch  # 解包batch
                    src = src.to(device)  # 源序列移到device
                    src_len = src_len.to(device)  # 源长度移到device
                    tgt = tgt.to(device)  # 目标序列移到device
                    tgt_len = tgt_len.to(device)  # 目标长度移到device
                    bos_ids = torch.full((tgt.shape[0], 1), tgt_vocab['<bos>'],
                                         dtype=torch.long, device=device)  # 构造<bos>列
                    dec_input = torch.cat([bos_ids, tgt[:, :-1]], dim=1)  # 构造decoder输入
                    pred = net(src, dec_input, src_len)  # 前向得到预测logits
                    loss = loss_fn(pred, tgt, tgt_len)  # 计算带mask的loss
                    val_loss_sum += loss.sum().item()  # 累加loss
                    val_token_count += int(tgt_len.sum().item())  # 累加有效token数

            val_loss_per_token = val_loss_sum / max(val_token_count, 1)  # 计算验证集平均loss/token
            # 计算验证集上的平均BLEU（k=2：unigram+bigram）
            val_bleu = evaluate_bleu_on_loader(
                net, val_loader, src_vocab, tgt_vocab, num_steps=src.shape[1], device=device, k=2
            )  # 调用BLEU评估函数
            print(f"==== Epoch {epoch + 1}/{num_epochs} Validation ====")  # 打印验证轮次
            print(f"Val loss/token: {val_loss_per_token:.4f}, Val BLEU@2: {val_bleu:.4f}")  # 打印验证loss与BLEU
            # 写入TensorBoard
            writer.add_scalar("Val/loss_per_token", val_loss_per_token, epoch + 1)  # 记录验证loss
            writer.add_scalar("Val/BLEU2", val_bleu, epoch + 1)  # 记录验证BLEU

            # 如果当前验证BLEU更高，则保存checkpoint和完整模型
            if val_loss_per_token < best_val_loss:  # 如果loss优于历史最优
                best_val_loss = val_loss_per_token  # 更新最优loss
                torch.save({
                    "model": net.state_dict(),  # 保存模型参数state_dict
                    "optimizer": optimizer.state_dict(),  # 保存优化器状态
                    "scheduler": scheduler.state_dict(),  # 保存学习率调度器状态
                    "epoch": epoch  # 保存当前epoch
                }, best_ckpt_path)  # 保存到checkpoint路径
                torch.save(net, best_model_path)  # 直接保存整个模型对象
                print(f"[Info] 新的最佳模型已保存，Val loss/token={best_val_loss:.4f}")  # 打印保存提示

    writer.close()  # 关闭TensorBoard日志记录器

    # ======== 9. 在测试集上评估最终BLEU（可选） ========
    print("[Info] 训练结束，加载最佳模型在测试集上评估...")  # 打印提示
    best_model = torch.load(best_model_path, map_location=device)  # 加载之前保存的最佳完整模型
    best_model.to(device)  # 将模型移到device
    test_bleu = evaluate_bleu_on_loader(
        best_model, test_loader, src_vocab, tgt_vocab, num_steps=src.shape[1], device=device, k=2
    )  # 在测试集上计算BLEU
    print(f"[Test] BLEU@2 on test set: {test_bleu:.4f}")  # 打印测试集BLEU


if __name__ == "__main__":  # 当本文件作为脚本直接运行时执行
    train_seq2seq()  # 调用训练函数启动训练流程