import torch
from torch import nn
from torch.nn import functional as F

NINF = -3.4 * 10 ** 38  # -INF
PAD = 1

class Code2VecEncoder(nn.Module):

    def __init__(self, word_vocab_size, path_vocab_size, embedding_dim, code_vec_dim, dropout):
        super(Code2VecEncoder, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, embedding_dim)
        self.path_embedding = nn.Embedding(path_vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3 * embedding_dim, code_vec_dim, bias=False)
        a = torch.nn.init.uniform_(torch.empty(code_vec_dim, 1, dtype=torch.float32, requires_grad=True))
        self.a = nn.parameter.Parameter(a, requires_grad=True)

    def forward(self, x_s, path, x_t):
        # x_s = path = x_t = [batch size, max length]
        batch_size = x_s.shape[0]
        
        # Embedding した結果のベクトルを concatenate
        embedded_x_s = self.word_embedding(x_s)
        embedded_path = self.path_embedding(path)
        embedded_x_t = self.word_embedding(x_t)
        # embedded_* = [batch size, max length, embedding dim]
        c = torch.cat((embedded_x_s, embedded_path, embedded_x_t), dim=2)
        # c = [batch size, max length, embedding dim * 3]

        # Fully Connected レイヤを通して combined context vector を計算
        cc = self.fc(self.dropout(c))
        cc = torch.tanh(cc)
        # cc = [batch size, max length, code vector dim]

        # Attention weight の計算
        aw = self.a.repeat(batch_size, 1, 1)  # aw = [batch size, code vector dim, 1]
        aw = torch.bmm(cc, aw)  # aw = [batch size, max length, 1]
        mask = ((x_s == PAD).float() * NINF).unsqueeze(2)  # mask = [batch size, max length, 1]
        aw = aw + mask
        aw = F.softmax(aw, dim=1)  # aw = [batch size, max length, 1]

        # context vector の attention 加重和をとって code vector を計算
        ca = torch.mul(cc, aw.expand_as(cc))  # ca = [batch size, max length, code vector dim]
        v = torch.sum(ca, dim=1)

        return v, aw


class Code2Vec(nn.Module):

    def __init__(self, word_vocab_size, path_vocab_size, label_vocab_size, embedding_dim, code_vec_dim, dropout):
        super(Code2Vec, self).__init__()
        self.encoder = Code2VecEncoder(word_vocab_size, path_vocab_size, embedding_dim, code_vec_dim, dropout)
        self.out = nn.Linear(code_vec_dim, label_vocab_size)

    def forward(self, x_s, path, x_t):
        v, aw = self.encoder(x_s, path, x_t)
        output = self.out(v)  # output = [batch size, output dim]
        return output, v
