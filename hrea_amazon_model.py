'''
ACTAAE模型构建
'''

import math, random, numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable as V

# TAAE
class TextEncoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 n_layers,           # LSTM的层数
                 bidirectional,      # 双向LSTM的设定
                 dropout,            # LSTM中的dropout设定
                 embedding,          # 词嵌入
                 input_size,
                 features = 200,     # 特征矩阵M中的特征
                 enc_dim  = 256      # 评论自编码表示向量的维度
                 ):
        super(TextEncoder, self).__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
                 input_size,
                 hidden_dim,
                 num_layers=n_layers,
                 bidirectional=bidirectional,
                 dropout=dropout,
                 batch_first=True)
        # --- 自注意力机制设定 ---
        self.enc_dim = enc_dim
        self.ws1 = nn.Linear(2 * hidden_dim, 500)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.ws2 = nn.Linear(500, features)
        self.softmax = nn.Softmax(dim=1)
        size = 2 * hidden_dim * features
        self.fc =  nn.Linear(270, enc_dim)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(720)
        self.flatten = nn.Flatten()

        self.cnn1 = nn.Conv2d(1, 20, 10, stride=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(20, 30, 10, stride=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 初始化
        self.__initit_lstm__(self.lstm)
        nn.init.xavier_uniform_(self.ws1.weight)
        nn.init.xavier_uniform_(self.ws2.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.cnn1.weight)
        nn.init.xavier_uniform_(self.cnn2.weight)

    def __initit_lstm__(self, lstm):
        for name, param in lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
    def forward(self, x, s):
        s = s.to('cpu')
        embeds = self.embedding(x)
        x_pack = pack_padded_sequence(embeds, s, batch_first=True, enforce_sorted=False)
        output, cell = self.lstm(x_pack)
        output2 = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
        H = output2[0]
        ws1 = self.tanh(self.ws1(H))
        A = self.softmax(self.ws2(ws1))
        M = torch.transpose(A, 1, 2) @ H

        # cnn构建
        M = torch.reshape(M, (M.shape[0],1 , M.shape[1], M.shape[2]))
        cnn1_out = self.cnn1(M)
        cnn1_pool = self.pool1(cnn1_out)
        cnn2_out = self.cnn2(cnn1_pool)
        cnn2_pool = self.pool2(cnn2_out)
        vec = self.flatten(cnn2_pool)
        drop = self.dropout(vec)
        # bnorm = self.batch_norm(drop)
        # enc = self.tanh(self.fc(drop))
        enc = self.fc(drop)
        return enc, embeds

# 自编码器构建
class TextAutoencoder(nn.Module):
    def __init__(self,
                 encoder,
                 embedding,
                 hidden_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 output_size):
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.n_layers = n_layers
        self.relu = nn.LeakyReLU()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.decoder = nn.LSTM(
                       input_size=embedding.weight.shape[1] + encoder.enc_dim,
                       hidden_size=hidden_dim,
                       num_layers=n_layers,
                       bidirectional=bidirectional,
                       dropout=dropout,
                       batch_first=True
                       )
        self.fc = nn.Linear(hidden_dim + encoder.enc_dim, 1000)
        self.output_layer = nn.Linear(1000, output_size)

         # 初始化
        self.__init_lstm__(self.decoder)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def __init_lstm__(self, lstm):
        for name, param in lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
    def forward(self, x, s, max_len, ratio=0.3):
        x2 = torch.clone(x)
        s = torch.tensor(s).to('cpu')
        context, cell = self.encoder(x2, s)     # context c

        # 用 <zero> 实现随机替换
        for i in torch.arange(x.shape[0]):
            end = int(sum(x[i] != 0))
            num = math.ceil(end * ratio)
            idx = random.sample(list(range(end)), num)
            x[i, idx] = 0

        # 构建解码器decoder的输入, 加入<start>作为开始标记符，并以一定概率用<zero>对词嵌入进行替换
        x[:, 1:] = x[:, 0:(x.shape[1] - 1)]  # append start token 2 <zero>
        x[:, 0]  = 0
        embeds = self.embedding(x)

        hidden2 = torch.unsqueeze(context, 1)
        hidden3 = hidden2.repeat(1,x.shape[1],1)
        decoder_input = torch.cat((embeds, hidden3),2)

        x_pack = pack_padded_sequence(decoder_input, s, batch_first=True, enforce_sorted=False)
        # decoder_output, cell = self.decoder(x_pack, cell)
        decoder_output, _ = self.decoder(x_pack)
        output2 = pad_packed_sequence(decoder_output, batch_first=True, total_length=max_len)
        lstm_output = output2[0]

        hidden4 = torch.unsqueeze(context, 1)
        hidden5 = hidden4.repeat(1,lstm_output.shape[1],1)
        hidden6 = torch.cat((lstm_output, hidden5),2)
        decoder_output = self.relu(self.fc(hidden6))
        outputs = self.output_layer(decoder_output)
        return outputs, context

    def get_encoder(self):
        return self.encoder

# 第二个encoder，即学生模型的构建
class Student1(nn.Module):
    def __init__(self, encoder, aspect_base, bank2_shape, enc_dim):
        super(Student1, self).__init__()
        self.encoder = encoder
        self.abase = aspect_base
        self.fc = nn.LazyLinear(1024)
        self.relu = nn.LeakyReLU()
        self.output = nn.LazyLinear(enc_dim)
        self.cnn1 = nn.Conv2d(1, 40, 2, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(40, 40, 2, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # self.W = nn.Linear(self.abase.shape[0], bank2_shape[0])
        self.tanh = nn.Tanh()
        self.generator = nn.Sequential(
          nn.LazyLinear(512),
          nn.LeakyReLU(),
          nn.LazyLinear(bank2_shape[0]),
          nn.Tanh()
        )

        # 初始化
        nn.init.xavier_uniform_(self.cnn1.weight)
        nn.init.xavier_uniform_(self.cnn2.weight)

    def forward(self, x, s):
        Se, E = self.encoder(x, s)
        self.selector = self.generator(torch.transpose(self.abase, 0, 1))

        A = E @ self.selector
        A = nn.Softmax(dim=2)(torch.transpose(A, 1, 2))
        M = A @ E
        M = torch.reshape(M, (M.shape[0], 1, M.shape[1], M.shape[2]))
        cnn1_out = self.cnn1(M)
        cnn1_pool = self.pool1(cnn1_out)
        cnn2_out = self.cnn2(cnn1_pool)
        cnn2_pool = self.pool2(cnn2_out)
        r = self.flatten(cnn2_pool)
        enc = self.output(self.relu(self.fc(r)))
        return enc, Se

    def get_aspect_base(self):
        return self.abase

    # 选择器的构建
    def get_bank(self):
        return torch.transpose(self.selector, 0, 1)

    def get_embeddings(self, vocab_size):
        len = np.array([vocab_size], dtype=int)
        idx = np.arange(vocab_size)
        idx = torch.from_numpy(idx)
        idx = torch.reshape(idx, (1, vocab_size))
        s = torch.tensor(len, dtype=torch.int64)
        _, E = self.encoder(idx, s)
        return E

# student 2 model
class Student2(nn.Module):
    def __init__(self, encoder, bank1, bank2_shape, enc_dim):
        super(Student2, self).__init__()
        self.encoder = encoder
        self.abase = bank1      # selector in student 1 as aspect base of student 2

        self.fc = nn.Linear(960, 200)
        self.relu = nn.LeakyReLU()
        self.output = nn.Linear(200, enc_dim)

        self.cnn1 = nn.Conv2d(1, 40, 2, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(40, 40, 2, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        # self.W = nn.Linear(self.bank1.shape[0], bank2_shape[0])

        self.generator = nn.Sequential(
            nn.LazyLinear(512),
            nn.LeakyReLU(),
            nn.LazyLinear(bank2_shape[0]),
            nn.Tanh()
        )
        # initialization
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.xavier_uniform_(self.cnn1.weight)
        nn.init.xavier_uniform_(self.cnn2.weight)

    def forward(self, x, s):
        Se, E = self.encoder(x, s)
        self.selector = self.generator(torch.transpose(self.abase, 0, 1))

        # scaled dot-product attention
        A = E @ self.selector
        A = nn.Softmax(dim=2)(torch.transpose(A, 1, 2))
        M = A @ E
        M = torch.reshape(M, (M.shape[0], 1, M.shape[1], M.shape[2]))
        cnn1_out = self.cnn1(M)
        cnn1_pool = self.pool1(cnn1_out)
        cnn2_out = self.cnn2(cnn1_pool)
        cnn2_pool = self.pool2(cnn2_out)
        r = self.flatten(cnn2_pool)
        enc = self.output(self.relu(self.fc(r)))

        return enc, Se

    # selector of bank in student1
    def get_bank(self):
        return torch.transpose(self.selector, 0, 1)
