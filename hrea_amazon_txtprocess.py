# 文本处理中的类

from collections import defaultdict
import re
from torch.utils.data import Dataset
import torch
import numpy as np
import nltk, operator

class Category:
    def __init__(self):
        self.idx_to_cate = dict()
        self.cate_to_idx = dict()

    def __len__(self):
        return len(self.idx_to_cate)

    def build(self, cate_list):
        cset = set(cate_list)
        self.cate_to_idx = {item: index for index, item in enumerate(cset)}
        self.idx_to_cate = {index: item for index, item in enumerate(cset)}
        clist = [self.cate_to_idx[item] for item in cate_list]
        return clist

    def convert_to_ids(self, cate_list):
        cset = set(cate_list)
        clist = [self.cate_to_idx[item] for item in cate_list]
        return clist

# id 0, 1 对应 "<zero>","<unk>"
# all zero padding give an embedding of all zero
# <mask> 被应用在降噪对抗自编码器中
class Vocab:
    def __init__(self, is_eng=True):
        self.idx_to_token = dict()
        self.token_to_idx = dict()
        self.is_eng = is_eng
        self.nouns_id = None  # id of nouns
        self.tokens = None  # unique tokens

    # 文本是一个列表，其中每个元素是一个句子
    def build(self, text, min_freq=1, reserved_tokens=None, noun_top=2000):
        token_freqs = defaultdict(int)
        noun_dic = defaultdict(int)
        nlist = ['NN', 'NNS', 'NNP', 'NNPS']

        for sentence in text:
            wlist = nltk.word_tokenize(sentence)
            postag = nltk.pos_tag(wlist)

            for item in postag:
                token_freqs[item[0]] += 1
                if item[1] in nlist:
                    noun_dic[item[0]] += 1

        uniq_tokens = ["<zero>","<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]

        sorted_nouns = sorted(noun_dic.items(), key=operator.itemgetter(1), reverse=True)
        nouns = sorted_nouns[0:noun_top]

        for id, token in enumerate(uniq_tokens):
            self.token_to_idx[token] = id
            self.idx_to_token[id] = token

        self.nouns_id = [self.token_to_idx[token] for token,_ in nouns]
        self.tokens = uniq_tokens

        return uniq_tokens, nouns

    # def __split__(self, sentence):
    #     if self.is_eng:
    #         sentence = sentence.lower()
    #         return filter(None, re.split("[ !\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r]", sentence))

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, 0)

    def convert_sentence_to_ids(self, sentence):
        # tokens = filter(None, re.split("[ !\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r]", sentence))
        tokens = nltk.word_tokenize(sentence)
        tlist = []
        nlist = []

        for token in tokens:    # 从词序列中排除 <unk> 
            id = self[token]
            if id == 2:
                continue
            else:
                tlist.append(id)
                if id in self.nouns_id:
                    nlist.append(1)
                else:
                    nlist.append(0)

        return tlist, nlist

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]

    def get_token_to_idx(self):
        return self.token_to_idx

    def get_idx_to_token(self):
        return self.idx_to_token

    def get_tokens(self):
        return self.tokens

class TxtDataset(Dataset):
    def __init__(self, pad_data, labels):
        self.labels = labels
        self.pad_data = pad_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.pad_data[idx,]
        y = self.labels[idx]
        return X, y

# 对于length可变性的设定
class TxtDataset2(Dataset):
    def __init__(self, pad_data, labels):
        self.labels = labels
        self.pad_data = pad_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.pad_data[idx,]
        y = self.labels[idx]
        length = len(X) - sum(X==0)
        return X, y, length

class TxtDataset3(Dataset):
    def __init__(self, pad_data, pad_data2, labels):
        self.labels = labels
        self.pad_data = pad_data
        self.pad_data2 = pad_data2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X1 = self.pad_data[idx,]
        X2 = self.pad_data2[idx,]
        y = self.labels[idx]
        len1 = len(X1) - sum(X1 == 0)
        len2 = len(X2) - sum(X2 == 0)
        return X1, X2, y, len1, len2


class TxtDataset4(Dataset):
    def __init__(self, pad_data, pad_data2, noun_data, noun_data2):
        # self.labels = labels
        self.pad_data = pad_data
        self.pad_data2  = pad_data2
        self.noun_data  = noun_data
        self.noun_data2 = noun_data2

    def __len__(self):
        return self.pad_data.shape[0]

    def __getitem__(self, idx):
        X1 = self.pad_data[idx,]
        X2 = self.pad_data2[idx,]
        # y = self.labels[idx]
        len1 = len(X1) - sum(X1 == 0)
        len2 = len(X2) - sum(X2 == 0)
        noun1 = self.noun_data[idx,]
        noun2 = self.noun_data2[idx,]

        return X1, X2, len1, len2, noun1, noun2

# 加载预训练词嵌入
class WordEmbeds():
    def __init__(self, word_index):
        self.word_index = word_index    # word->id的字典

    def load(self, embed_file, embed_dim):
        num = 0
        vocab_size = len(self.word_index)
        embeddings_index = {}
        with open(embed_file, encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size, embed_dim),dtype=np.float32)
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                num += 1

        # 设定<mask>的词嵌入. 对于所有词向量进行means操作
        embedding_matrix[1, ] = np.mean(embedding_matrix[2:, ], axis=0)
        return embedding_matrix, num

    # 构建由名词组成的方面库
    def build_aspect_base(self, embed_file, nouns, embed_dim):
        num = 0
        embeddings_index = {}
        with open(embed_file, encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                if word not in self.word_index: continue
                token_id = self.word_index[word]
                embeddings_index[token_id] = coefs

        abase = np.zeros((len(nouns), embed_dim),dtype=np.float32)
        for i, token_id in enumerate(nouns):
            embedding_vector = embeddings_index.get(token_id)
            if embedding_vector is not None:
                abase[i] = embedding_vector
            # else:
            #     num += 1
        return abase
