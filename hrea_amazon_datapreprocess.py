'''
对Amazon数据集进行实验
该代码文件为数据预处理过程
将保存内容：数据、词典、类别标签与词嵌入矩阵
'''

import pickle
import os
import torch
import json
import numpy as np
import nltk

from hrea_amazon_txtprocess import Vocab, Category, TxtDataset2, TxtDataset4, WordEmbeds

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# 1. 超参数的选择
review_file   = 'amazon-review-100k.json'
embed_file    = 'amazon-embed.txt'
# review_file   = 'amazon-yelp-review-1m.json'
# embed_file    = 'amazon-1M-embeds-100.txt'
embedding_dim = 50        # 预训练词嵌入矩阵的维数
word_freq     = 35
num_worker    = 15
max_seq_length= 400

print(os.path.basename(__file__))

# 1. 构建数据集
fin         = open(review_file)
lines       = fin.readlines()
cate_list   = list()
review_list = list()

# 1.1 读取文件，建立词典和类别
for line in lines:
    dic = json.loads(line)
    cate_list.append(dic['category'])
    review_list.append(dic['text'].lower())
fin.close()

vocab = Vocab()
uniq_tokens, noun_list = vocab.build(review_list,min_freq=word_freq)
vocab_size = len(uniq_tokens)
print(f"Vocab: {vocab_size} tokens, {len(noun_list)} nouns")

data = []
noun_data = []
cate_list2 = []

for line, cate in zip(review_list, cate_list):
    ids, ns = vocab.convert_sentence_to_ids(line)
    if sum(ids) == 0: continue
    data.append(ids)
    noun_data.append(ns)
    cate_list2.append(cate)

category = Category()
clist = torch.tensor(category.build(cate_list2), dtype=int)
seq_len = max([len(line) for line in data])
print(f"max sequence length: {seq_len}")
seq_len = seq_len if seq_len < max_seq_length else max_seq_length

# 1.2 建立一个填充矩阵
pad_data = np.zeros((len(data), seq_len), dtype=int)
pad_noun = np.zeros((len(data), seq_len), dtype=int)

for i, line in enumerate(zip(data,noun_data)):
    if len(line[0]) < seq_len:
        size = len(line[0])
        pad_data[i, 0:size] = line[0]
        pad_noun[i,0:size] = line[1]
    else:
        pad_data[i, :] = line[0][0:seq_len]
        pad_noun[i, :] = line[1][0:seq_len]

pad_data = torch.tensor(pad_data)
pad_noun = torch.tensor(pad_noun)
print(f"pad_data shape: {pad_data.shape}")

shuffle_indices = np.random.permutation(np.arange(len(data)))
pad_data1 = pad_data[shuffle_indices]
noun_data1 = pad_noun[shuffle_indices]
clist = clist[shuffle_indices]

shuffle_indices = np.random.permutation(shuffle_indices)
pad_data2 = pad_data[shuffle_indices]
pad_noun2 = pad_noun[shuffle_indices]

# 1.3 构建数据加载器
dataset  = TxtDataset2(pad_data1, clist)                          # for encoder use
dataset2 = TxtDataset4(pad_data1, pad_data2, pad_noun, pad_noun2) # for hiercluster use

# 1.4 词嵌入
word_embeds = WordEmbeds(vocab.get_token_to_idx())
weight_matrix,_ = word_embeds.load(embed_file, embedding_dim)

# 1.5 保存文件
filename = 'dataset.pl'
outfile = open(filename,'wb')
pickle.dump(dataset, outfile)
outfile.close()

filename = 'dataset2.pl'
outfile = open(filename,'wb')
pickle.dump(dataset2, outfile)
outfile.close()

filename = 'vocab.pl'
outfile = open(filename,'wb')
pickle.dump(vocab,outfile)
outfile.close()

filename = 'category.pl'
outfile = open(filename,'wb')
pickle.dump(category,outfile)
outfile.close()

filename = 'weight_matrix.pl'
outfile = open(filename,'wb')
pickle.dump(weight_matrix,outfile)
outfile.close()

print('Done!')
