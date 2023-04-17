# 基于学生-教师学习训练第二个自编码器（学生模型）

import numpy as np
import os, math, time
import torch
import pickle
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import random_split
from hrea_amazon_model import Student1
from hrea_amazon_txtprocess import WordEmbeds

print(os.path.basename(__file__))

# 1. 超参数的选择
embed_file = 'amazon-embed.txt'
encoder_name = 'encoder.pth'
filename = 'encoder_parameters.pl'
with open(filename, 'rb') as f:
    parameters_dic = pickle.load(f)

epochs = 40
lr = 0.002
alpha = 1
batch_size = 300
bank_size = 50  # selector size
num_workers = 12
embed_dim = parameters_dic['embedding_dim']
enc_dim = parameters_dic['encoder_enc_dim']
word_freq = parameters_dic['word_freq']
max_seq_length = parameters_dic['max_seq_length']
train_test_ratio = 0.9

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

# 2. 设置种子seed和GPU
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(100)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
print("Using device", device)

# 3. 文件与数据集的读取
with open('vocab.pl', 'rb') as f:
    vocab = pickle.load(f)

with open('category.pl', 'rb') as f:
    category = pickle.load(f)

with open('dataset2.pl', 'rb') as f:
    dataset = pickle.load(f)

nouns = vocab.nouns_id
dic = vocab.get_idx_to_token()
vocab_size = len(dic)
print(f'vocabulary size: {vocab_size}')

# 包含id的名字, 包含了词向量的方面库
word_embeds = WordEmbeds(vocab.get_token_to_idx())
abase = word_embeds.build_aspect_base(embed_file, nouns, embed_dim) # aspect base
abase = torch.tensor(abase).to(device)

def get_dataloader(dataset, train_test_ratio):
    train_size = math.floor(len(dataset) * train_test_ratio)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader

# 4. 加载自编码器
daae = torch.load(encoder_name)  # 预训练的自编码器
encoder = daae.module.encoder
encoder = nn.DataParallel(encoder.to(device))

for param in encoder.parameters():  # 将自编码器部分设置为不可训练
    param.requires_grad = False

model = Student1(encoder.to(device), abase, (bank_size, embed_dim), enc_dim).to(device)

# 5. 损失函数 loss = dist(Se,Sp) + Sp*Sn
pdist = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

def myloss(Se, Sp, Sn):
    a = torch.mean(pdist(Se, Sp))
    b = torch.mean(torch.maximum(torch.tensor(0), torch.sum(Sp * Sn, dim=1)))
    loss = a + alpha * torch.maximum(torch.tensor(0), b)
    return loss

# 6. 训练
def train_loop(train_dataloader, model, loss_fn, optimizer):
    total_loss = 0
    num_batches = len(train_dataloader)

    for Xp, Xn, lp, ln, nounp, nounn in train_dataloader:  # p 为正实例, n 为负实例
        lp = torch.tensor(lp, dtype=torch.int32)
        ln = torch.tensor(ln, dtype=torch.int32)
        Sp, Se = model(Xp.to(device), lp)
        Sn, _ = model(Xn.to(device), ln)
        loss = loss_fn(Se, Sp, Sn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / num_batches


def test_loop(dataloader, model, loss_fn, is_test):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X1, X2, s1, s2, _, _ in dataloader:
            Sp_l2, Se = model(X1.to(device), s1)
            Sn_l2, _  = model(X2.to(device), s2)

            test_loss += loss_fn(Se, Sp_l2, Sn_l2).item()

    test_loss /= num_batches
    return test_loss

optimizer = torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.5, 0.99))

for t in range(epochs):
    start_time = time.time()
    train_loader, test_loader = get_dataloader(dataset, train_test_ratio)
    model.train()  # set status in training
    train_loss = train_loop(train_loader, model, myloss, optimizer)
    run_time = time.time() - start_time
    model.eval()  # set status in evaluation
    valid_loss = test_loop(test_loader, model, myloss, False)
    print(
        f"epoch {t + 1:2d}/{epochs}, train loss: {train_loss:0.3f}, valid loss:{valid_loss:0.3f}, time:{run_time:0.2f}")

# 7. 保存方面
with open(f'pkg_level1_bank{bank_size}.model', 'wb') as f:
    pickle.dump(model, f)

with open('bank1.out', 'wb') as f:
    bank = model.get_bank()
    bank = bank.cpu().detach().numpy()
    pickle.dump(bank, f)
