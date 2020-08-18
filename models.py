import torch
from torch import nn
from torch.nn import functional as F

NINF = -3.4 * 10 ** 38  # -INF
PAD = 1

class Code2Vec(nn.Module):

    def __init__(self, words_dim, paths_dim, embedding_dim, output_dim, dropout):
        super(Code2Vec, self).__init__()
        self.node_embedding = nn.Embedding(words_dim, embedding_dim)
        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)
        self.W = nn.Parameter(torch.randn(1, embedding_dim, 3*embedding_dim))
        self.a = nn.Parameter(torch.randn(1, embedding_dim, 1))
        self.out = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_s, path, x_t):
        # x_s = path = x_t = [batch size, max length]
        batch_size = x_s.shape[0]
        
        # Embedding した結果のベクトルを concatenate
        embedded_x_s = self.node_embedding(x_s)
        embedded_path = self.path_embedding(path)
        embedded_x_t = self.node_embedding(x_t)
        # embedded_* = [batch size, max length, embedding dim]
        c = self.dropout(torch.cat((embedded_x_s, embedded_path, embedded_x_t), dim=2))
        # c = [batch size, max length, embedding dim * 3]

        # Fully Connected レイヤを通して combined context vector を計算
        W = self.W.repeat(batch_size, 1, 1)  # batch_size 分だけ複製する
        # W = [batch size, embedding dim, embedding dim * 3]
        c = c.permute(0, 2, 1)  # 積の計算順序の都合上、テンソルの次元を入れ替えてから計算して戻す
        c = torch.tanh(torch.bmm(W, c))
        c = c.permute(0, 2, 1)
        # c = [batch size, max length, embedding dim]

        # Attention weight の計算
        a = self.a.repeat(batch_size, 1, 1)  # batch_size 分だけ複製する a = [batch size, embedding dim, 1]
        a = torch.bmm(c, a).squeeze(2)  # a = [batch size, max length]
        a = a + ((x_s == PAD).float() * NINF)  # <pad> 部分の softmax 結果を0にするために，事前に-INFにしておく
        a = F.softmax(a, dim=1).unsqueeze(2)
        # a = [batch size, max length, 1]

        # Code vector の計算
        c = c.permute(0, 2, 1)
        # c = [batch size, embedding dim, max length]
        v = torch.bmm(c, a).squeeze(2)
        # v = [batch size, embedding dim]

        # ここまでで code vector (v) が計算できた
        # この後、訓練時は出力層のノード数に合わせて線形変換する必要がある
        output = self.out(v)  # output = [batch size, output dim]
        return output, v
