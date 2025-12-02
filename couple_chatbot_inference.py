import torch
from couple_chatbot_dataloader import truncate_pad    # 直接复用你的 padding 函数
from couple_chatbot_seq2seq import Seq2SeqEncoder, Seq2SeqAttentionDecoder, Seq2Seq
import torch
from couple_chatbot_dataloader import truncate_pad


# -----------------------------------------------------------
#   1. 采样策略函数
# -----------------------------------------------------------

def sample_top_k(probs, k=40):
    """Top-K 采样"""
    topk_probs, topk_idx = torch.topk(probs, k)
    topk_probs = topk_probs / topk_probs.sum()
    sampled = torch.multinomial(topk_probs, 1).item()
    return topk_idx[sampled].item()


def sample_top_p(probs, p=0.9):
    """Top-p (nucleus) 采样"""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    # 找到累计概率超过 p 的位置
    cutoff = torch.where(cumulative > p)[0]
    if len(cutoff) == 0:
        cutoff_idx = len(sorted_probs) - 1
    else:
        cutoff_idx = cutoff[0].item()

    # 使用前 cutoff_idx + 1 的 token
    filtered_probs = sorted_probs[:cutoff_idx + 1]
    filtered_probs = filtered_probs / filtered_probs.sum()

    sampled = torch.multinomial(filtered_probs, 1).item()
    return sorted_idx[sampled].item()


def sample_with_temperature(probs, temperature=1.0):
    """Temperature 采样"""
    logits = torch.log(probs + 1e-12) / temperature
    adjusted_probs = torch.softmax(logits, dim=-1)
    sampled = torch.multinomial(adjusted_probs, 1).item()
    return sampled


# -----------------------------------------------------------
#   主推理函数：整合 4 种策略
# -----------------------------------------------------------

def predict_reply(net,
                  text,
                  src_vocab,
                  tgt_vocab,
                  device="cuda",
                  num_steps=50,
                  decode_strategy="top_p",     # 可选：greedy, top_k, top_p, temperature
                  top_k=40,
                  top_p=0.9,
                  temperature=0.8):
    """
    主推理函数（支持 greedy / top-k / top-p / temperature）
    """

    # ---------------------- 编码输入句子 ----------------------
    src_tokens = list(text)
    src_ids_no_pad = src_vocab[src_tokens] + [src_vocab["<eos>"]]
    valid_len = min(len(src_ids_no_pad), num_steps)

    src_ids = truncate_pad(src_ids_no_pad, num_steps, src_vocab["<pad>"])
    enc_X = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    enc_valid_len = torch.tensor([valid_len], dtype=torch.int32, device=device)

    net.eval()
    with torch.no_grad():
        enc_outputs = net.encoder(enc_X, enc_valid_len)
        dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    # ---------------------- 解码 ----------------------
    bos_id = tgt_vocab["<bos>"]
    dec_input = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    output_ids = []

    for _ in range(num_steps):
        with torch.no_grad():
            Y, dec_state = net.decoder(dec_input, dec_state)

        logits = Y[0, 0]
        probs = torch.softmax(logits, dim=-1)

        # 选择策略
        if decode_strategy == "greedy":
            next_id = probs.argmax().item()

        elif decode_strategy == "top_k":
            next_id = sample_top_k(probs, k=top_k)

        elif decode_strategy == "top_p":
            next_id = sample_top_p(probs, p=top_p)

        elif decode_strategy == "temperature":
            next_id = sample_with_temperature(probs, temperature=temperature)

        else:
            raise ValueError(f"未知策略：{decode_strategy}")

        if next_id == tgt_vocab["<eos>"]:
            break

        output_ids.append(next_id)
        dec_input = torch.tensor([[next_id]], dtype=torch.long, device=device)

    # ---------------------- 转回文本 ----------------------
    reply_chars = tgt_vocab.to_tokens(output_ids)
    reply_text = "".join(reply_chars)

    return reply_text



# -----------------------------------------------------
# 2. 示例调用（你可以像 main 一样运行本文件）
# -----------------------------------------------------
if __name__ == "__main__":
    print("=== 加载词表... ===")
    # 你需要从训练脚本保存 vocab 再加载，这里假设你保存了 vocab
    # 如果你没有保存 vocab，我可以教你如何保存 & 加载
    vocab_data = torch.load("vocab_dict.pth", weights_only=False)
    src_vocab = vocab_data["src_vocab"]
    tgt_vocab = vocab_data["tgt_vocab"]
    me_name = vocab_data["input_name"]
    other_name = vocab_data["output_name"]

    print(f"=== 加载昵称成功：我方 = {me_name}, 对方 = {other_name} ===")

    print("=== 构建模型骨架... ===")
    embed_size = 256  # ← 和你训练时的一致
    num_hiddens = 512  # ← 和你训练时的一致
    num_layers = 2  # 报错里能看到 l0 / l1，就是 2 层
    dropout = 0.1  # 用你训练时的值，0.1/0.2 都不会影响加载

    encoder = Seq2SeqEncoder(len(src_vocab), embed_size,
                             num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size,
                             num_hiddens, num_layers, dropout)
    net = Seq2Seq(encoder, decoder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)

    print("=== 加载训练好的 checkpoint... ===")
    checkpoint_path = "best_seq2seq_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model"])

    print("=== 模型加载完成，可以开始对话 ===")

    while True:
        user_text = input(f"\n{me_name}：")
        if user_text.lower() in ["quit", "exit"]:
            break

        reply = predict_reply(net, user_text, src_vocab, tgt_vocab, device)
        print(f"{other_name}：", reply)
