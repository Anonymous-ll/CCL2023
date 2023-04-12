'''
amazon dataset
Training a denoising text adversarial autoencoder.
'''

import pickle
import time
import os
import random

import torch
import json
import math
import numpy as np

from hrea_amazon_txtprocess import Category, TxtDataset2, WordEmbeds
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import random_split
from hrea_amazon_model import TextAutoencoder, TextEncoder


# 1. hyper-parameters
g_lr          = 0.0008
d_lr          = 0.002
epochs        = 20
batch_size    = 180
embedding_dim = 50        # dim of pretrained word embeds
n_layers      = 2
hidden_dim    = 512
dropout       = 0.5
word_freq     = 35
lam1          = 0.6
lam2          = 0.5
num_worker    = 15
max_seq_length   = 400
train_test_ratio = 0.8
train_vali_ratio = 0.9
encoder_enc_dim  = 100        # dimentions of encoded reivew
word_embeds_trainable = False

print(os.path.basename(__file__))

parameters_dic={'embedding_dim':embedding_dim,
                'word_freq':word_freq,
                'max_seq_length':max_seq_length,
                'encoder_enc_dim':encoder_enc_dim}

filename = 'encoder_parameters.pl'
outfile = open(filename,'wb')
pickle.dump(parameters_dic,outfile) # for hiercluster
outfile.close()

# 2. set the seed and GPU
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(422)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
print("Using device", device)

# 3. dataset
filename = 'dataset.pl'
outfile = open(filename,'rb')
dataset = pickle.load(outfile)
outfile.close()

filename = 'vocab.pl'
outfile = open(filename,'rb')
vocab = pickle.load(outfile)
outfile.close()

filename = 'category.pl'
outfile = open(filename,'rb')
category = pickle.load(outfile)
outfile.close()

filename = 'weight_matrix.pl'
outfile = open(filename,'rb')
weight_matrix = pickle.load(outfile)
outfile.close()

noun_list = vocab.nouns_id
uniq_tokens = vocab.get_tokens()
vocab_size = len(uniq_tokens)
print(f"Vocab: {vocab_size} tokens, {len(noun_list)} nouns")

def get_dataloader(dataset, train_test_ratio, num_workers):
    train_size = math.floor(len(dataset) * train_test_ratio)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader


# 4. model
# 4.1 building embed layer
num_embeddings, _ = weight_matrix.shape
emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
emb.weight = torch.nn.Parameter(torch.from_numpy(weight_matrix))
emb.weight.requires_grad = word_embeds_trainable
input_size = weight_matrix.shape[1]

#4.2 building autoencoder
encoder = TextEncoder(
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    dropout=dropout,
                    bidirectional=True,
                    embedding=emb,                  # word embeddings
                    input_size=input_size,
                    enc_dim=encoder_enc_dim).to(device)


# text autoencoder including a decoder
model = TextAutoencoder(encoder,
                        embedding=emb,
                        hidden_dim=hidden_dim,
                        n_layers=n_layers,
                        bidirectional=False,
                        dropout=0.5,
                        output_size=vocab_size)

model = nn.DataParallel(model.to(device))

# 4.3 discriminator
discriminator = nn.Sequential(
    nn.Linear(in_features=encoder_enc_dim, out_features=128),
    nn.LeakyReLU(),
    nn.Linear(in_features=128, out_features=64),
    nn.LeakyReLU(),
    nn.Linear(in_features=64, out_features=1),
    nn.Sigmoid()
    )
discriminator = nn.DataParallel(discriminator.to(device))

# loss function of discriminator
bceloss = nn.BCELoss()
def discriminator_loss(real_output, fake_output):
    real_loss = bceloss(real_output, torch.ones(real_output.shape). to(device))
    fake_loss = bceloss(fake_output, torch.zeros(fake_output.shape).to(device))
    total_loss = (1-lam1) * real_loss + lam1 * fake_loss
    return total_loss

def generator_loss(fake_output):
    fake_loss = bceloss(fake_output, torch.ones(fake_output.shape).to(device))
    return fake_loss

# 5. training
# (1) initialization
def xavier_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            bound = math.sqrt(6)/math.sqrt(param.shape[0]+param.shape[1])
            param.data.uniform_(-bound, bound)

xavier_init(discriminator)


# (2) training
loss_fn   = nn.CrossEntropyLoss()
optimizerG = torch.optim.Adam(model.parameters(), lr=g_lr, betas=(0.5, 0.995))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
# optimizerD = torch.optim.RMSprop(discriminator.parameters(), lr=d_lr, momentum=0.9)

def train_loop(train_dataloader, model, loss_fn):
    gen_loss = 0
    dis_loss = 0

    for x, _, s in train_dataloader:
        mu, sigma = 0, 2
        z = np.random.normal(loc=mu, scale=sigma, size=(len(x), encoder_enc_dim))
        z = torch.tensor(z, dtype=torch.float).to(device)
        max_len = max(s.numpy())

        # update generator
        model.zero_grad()
        pred, encoded_doc = model(x.to(device), s, max_len)  # context is also encoded doc
        pred = torch.transpose(pred, 1, 2)
        f_output = discriminator(encoded_doc)
        rec_loss = loss_fn(pred, x[:,0:max_len].to(device))  # reconstructed loss
        gloss = lam2 * rec_loss + (1-lam2) * generator_loss(f_output)
        gloss.backward()
        optimizerG.step()
        gen_loss += gloss.item()

        # update discriminator
        discriminator.zero_grad()
        r_output = discriminator(z)
        dloss = discriminator_loss(r_output, f_output.detach())
        dloss.backward()
        optimizerD.step()
        dis_loss += dloss.item()

    return gen_loss, dis_loss

def test_loop(dataloader, model, loss_fn):
    test_loss = 0

    with torch.no_grad():
        for x, _, s in dataloader:
            max_len = max(s.numpy())
            pred, encoded_doc = model(x.to(device), s, max_len)
            pred = torch.transpose(pred, 1, 2)
            f_output = discriminator(encoded_doc)
            rec_loss = loss_fn(pred, x[:,0:max_len].to(device)).item()
            test_loss += (1-lam2) * rec_loss + lam2 * generator_loss(f_output)

    return test_loss

import torch
torch.cuda.empty_cache()

for t in range(epochs):
    train_loader, _ = get_dataloader(dataset, train_test_ratio, num_worker)
    start_time = time.time()
    model.train()
    gen_loss, dis_loss = train_loop(train_loader, model, loss_fn)
    run_time = time.time() - start_time
    # model.eval()
    # valid_loss = test_loop(test_loader, model, loss_fn)

    print(f"epoch {t+1:2d}/{epochs}, gen loss: {gen_loss:0.4f}, dis loss: {dis_loss:0.4f}, time:{run_time:0.2f}")

torch.save(model, f'encoder.pth')
print("Training Done!")
