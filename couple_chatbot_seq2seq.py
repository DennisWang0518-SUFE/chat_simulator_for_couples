import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================
# 1. 编码器：和你现在的一样（单向 GRU）
# =========================================
class Seq2SeqEncoder(nn.Module):
    """用于 seq2seq 的 RNN 编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # 词嵌入层
        self.rnn = nn.GRU(
            input_size=embed_size,       # 输入维度 = 词向量维度
            hidden_size=num_hiddens,     # 隐状态维度
            num_layers=num_layers,       # 堆叠层数
            dropout=dropout,             # 层间 dropout
            bidirectional=False          # 单向 GRU
        )

    def forward(self, X, valid_len):
        """
        X: (batch_size, num_steps)       已经 pad 好的输入
        valid_len: (batch_size,)         每个样本的有效长度（不含 pad）
        """
        # 嵌入： (batch, num_steps, embed_size)
        X = self.embedding(X)

        # 调整维度给 GRU： (num_steps, batch, embed_size)
        X = X.permute(1, 0, 2)

        # GRU 前向：
        # enc_outputs: (num_steps, batch, num_hiddens)  每个时间步的输出
        # enc_state:   (num_layers, batch, num_hiddens) 最后一个时间步的隐状态
        enc_outputs, enc_state = self.rnn(X)

        # 不在这里 mask，mask 在 attention 里做
        return enc_outputs, enc_state


# =========================================
# 2. 最简单的 dot-product Attention（Luong-style）
# =========================================
class DotProductAttention(nn.Module):
    """
    最简单的点积注意力：
    score(h_t, h_s) = h_s · h_t
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # 可选的 dropout

    def forward(self, query, keys, values, valid_len):
        """
        query:  (batch, 1, hidden)              当前解码步的隐状态
        keys:   (batch, src_len, hidden)        encoder 所有时间步的输出
        values: (batch, src_len, hidden)        一般和 keys 一样
        valid_len: (batch,)                     每个样本的有效长度
        """
        # 计算点积相关性 scores: (batch, src_len)
        # keys:   (batch, src_len, hidden)
        # query:  (batch, 1, hidden) → (batch, hidden, 1)
        # bmm:    (batch, src_len, hidden) x (batch, hidden, 1)
        scores = torch.bmm(keys, query.transpose(1, 2)).squeeze(-1)

        # 对 padding 部分做 mask
        if valid_len is not None:
            max_len = keys.size(1)  # src_len
            # positions: (1, max_len) = [0,1,2,...]
            positions = torch.arange(max_len, device=keys.device).unsqueeze(0)
            # valid_len: (batch, 1)
            mask = positions < valid_len.unsqueeze(1)
            # 超出有效长度的部分设为 -1e9
            scores = scores.masked_fill(~mask, -1e9)

        # softmax 得到注意力权重 α: (batch, src_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和得到上下文向量 context: (batch, hidden)
        # attn_weights: (batch, 1, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)

        return context, attn_weights


# =========================================
# 3. 带“轻量注意力”的解码器
# =========================================
class Seq2SeqAttentionDecoder(nn.Module):
    """使用 dot-product attention 的 GRU 解码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0.0):
        super().__init__()

        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)      # 词嵌入
        self.attention = DotProductAttention(dropout=dropout)      # 简单 dot attention

        # RNN 输入 = 当前词向量 + context（hidden）
        self.rnn = nn.GRU(
            input_size=embed_size + num_hiddens,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            dropout=dropout
        )

        self.dense = nn.Linear(num_hiddens, vocab_size)            # 输出到词表

    def init_state(self, enc_outputs, enc_valid_len):
        """
        enc_outputs: (enc_all_outputs, enc_state)
            enc_all_outputs: (src_len, batch, hidden)
            enc_state:       (num_layers, batch, hidden)
        enc_valid_len: (batch,)
        """
        enc_all_outputs, enc_state = enc_outputs
        # 调成 (batch, src_len, hidden) 方便做 attention
        enc_all_outputs = enc_all_outputs.permute(1, 0, 2)
        # state 统一打包成一个 tuple
        return (enc_all_outputs, enc_state, enc_valid_len)

    def forward(self, X, state):
        """
        X: (batch, tgt_len)        解码器输入序列（训练时：<bos> + 右移后的 target）
        state: (enc_outputs, dec_state, enc_valid_len)
        返回：
            outputs: (batch, tgt_len, vocab_size)
            new_state: 更新后的 state
        """
        enc_outputs, dec_state, enc_valid_len = state  # 解包

        # 1. embedding: (batch, tgt_len, embed)
        X = self.embedding(X)
        # 2. 调整成 (tgt_len, batch, embed) 以便逐时间步处理
        X = X.permute(1, 0, 2)

        outputs = []  # 存每个时间步的 logits

        # 循环每个时间步（和你原来 decoder 的写法类似）
        for x_t in X:  # x_t: (batch, embed)
            # x_t: (batch, 1, embed)
            x_t = x_t.unsqueeze(1)

            # 当前 query = 最后一层decoder隐状态: (batch, hidden)
            query = dec_state[-1].unsqueeze(1)  # (batch, 1, hidden)

            # enc_outputs: (batch, src_len, hidden) 做 keys / values
            context, _ = self.attention(query, enc_outputs, enc_outputs, enc_valid_len)
            # context: (batch, hidden) → (batch, 1, hidden)
            context = context.unsqueeze(1)

            # 拼接当前 token embedding 和 context: (batch, 1, embed+hidden)
            rnn_input = torch.cat((x_t, context), dim=-1)

            # 调成 (1, batch, embed+hidden) 送入 GRU
            rnn_input = rnn_input.permute(1, 0, 2)

            # GRU 一步前向
            rnn_output, dec_state = self.rnn(rnn_input, dec_state)
            # rnn_output: (1, batch, hidden) → (batch, hidden)
            out_t = rnn_output.squeeze(0)

            # 映射到词表维度： (batch, vocab_size)
            logits = self.dense(out_t)
            outputs.append(logits)

        # 拼回 (tgt_len, batch, vocab) → (batch, tgt_len, vocab)
        outputs = torch.stack(outputs, dim=0).permute(1, 0, 2)

        return outputs, (enc_outputs, dec_state, enc_valid_len)


# =========================================
# 4. Seq2Seq 封装：接口保持不变
# =========================================
class Seq2Seq(nn.Module):
    """Encoder + Attention Decoder 封装"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_len):
        """
        enc_X: (batch, src_len)
        dec_X: (batch, tgt_len)
        enc_valid_len: (batch,)
        """
        enc_outputs = self.encoder(enc_X, enc_valid_len)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_len)
        outputs, _ = self.decoder(dec_X, dec_state)
        return outputs