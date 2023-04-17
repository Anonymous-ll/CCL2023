# 方面提取

import numpy as np
import pickle
from hrea_amazon_txtprocess import WordEmbeds

topn = 30

with open('bank1.out', 'rb') as f:
    bank = pickle.load(f)

with open('vocab.pl', 'rb') as f:
    vocab = pickle.load(f)
bank_size = bank.shape[0]   # 选择器的大小设定
model_name = f'pkg_level1_bank{bank_size}.model'
embed_file = 'amazon-embed.txt'
filename   = 'encoder_parameters.pl'

with open(filename, 'rb') as f:
    parameters_dic = pickle.load(f)

embed_dim = parameters_dic['embedding_dim']

nouns = vocab.nouns_id
word_embeds = WordEmbeds(vocab.get_token_to_idx())
E = word_embeds.build_aspect_base(embed_file, nouns, embed_dim)  # aspects base

idx2token = vocab.get_idx_to_token()
vocab_size = len(idx2token)

idx_word = {}                       # 名词索引对应
for i in range(len(nouns)):
    idx_word[i] = idx2token[nouns[i]]

M = np.matmul(bank, np.transpose(E))

# 找到方面簇中与质心最相近的前n个词
def find(vec, idx_word, topn):
    idx = np.argsort(vec)[::-1][:topn]
    return ([f'{idx_word[id]}:{vec[id]:0.1f}' for id in idx])

for i in np.arange(M.shape[0]):
    words = find(M[i], idx_word, topn)
    print(words)

np.savetxt('bank1.txt', bank, fmt='%.4e', delimiter=',', encoding='utf-8')
