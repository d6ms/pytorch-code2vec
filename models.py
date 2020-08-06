import torch
from torch import nn
from torch.nn import functional as F


class Code2Vec(nn.Module):

    def __init__(self, nodes_dim, paths_dim, embedding_dim, output_dim, dropout):
        super(Code2Vec, self).__init__()
        self.node_embedding = nn.Embedding(nodes_dim, embedding_dim)
        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)
        self.W = nn.Parameter(torch.randn(1, embedding_dim, 3*embedding_dim))
        self.a = nn.Parameter(torch.randn(1, embedding_dim, 1))
        self.out = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, starts, paths, ends):
        #starts = paths = ends = [batch size, max length]
        
        # Embedding した結果のベクトルを concatenate
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        # embedded_* = [batch size, max length, embedding dim]
        c = self.dropout(torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2))
        # c = [batch size, max length, embedding dim * 3]

        # Fully Connected レイヤを通して combined context vector を計算
        W = self.W.repeat(starts.shape[0], 1, 1)  # batch_size 分だけ複製する
        # W = [batch size, embedding dim, embedding dim * 3]
        c = c.permute(0, 2, 1)  # 積の計算順序の都合上、テンソルの次元を入れ替えてから計算して戻す
        x = torch.tanh(torch.bmm(W, c))
        x = x.permute(0, 2, 1)
        # x = [batch size, max length, embedding dim]

        # Attention weight の計算
        a = self.a.repeat(starts.shape[0], 1, 1)  # batch_size 分だけ複製する
        # a = [batch size, embedding dim, 1]
        z = F.softmax(torch.bmm(x, a).squeeze(2), dim=1).unsqueeze(2)
        # z = [batch size, max length, 1]

        # Code vector の計算
        x = x.permute(0, 2, 1)
        # x = [batch size, embedding dim, max length]
        v = torch.bmm(x, z).squeeze(2)
        # v = [batch size, embedding dim]

        # ここまでで code vector (v) が計算できた
        # この後、訓練時は出力層のノード数に合わせて線形変換する必要がある
        return self.out(v)  # output = [batch size, output dim]
