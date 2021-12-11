import numpy as np
from torch import nn
from torch import Tensor
import csv
import torch
import json
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer


#load sentences as list of list of indexes using word2id json dict

def load_data(Word2Idx, path):
    x = []
    y = []
    stemmer = SnowballStemmer("english")
    wnl = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    with open(path, 'r') as f: 
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        for i, row in enumerate(rdr): 
            line = ' '.join(row[1:]).lower()
            # x.append(list(map(lambda x: Word2Idx[stemmer.stem(x)] if stemmer.stem(x) in Word2Idx.keys() else 0, tokenizer.tokenize(line))))
            # x.append(list(map(lambda x: Word2Idx[wnl.lemmatize(x, pos='v')] if wnl.lemmatize(x, pos='v') in Word2Idx.keys() else 0, tokenizer.tokenize(line))))
            token_ids = []
            for word in tokenizer.tokenize(line):    
                v_lem = wnl.lemmatize(word, pos='v')
                a_lem = wnl.lemmatize(word, pos='a')
                n_lem = wnl.lemmatize(word, pos='n')
                s_lem = stemmer.stem(word)
                if (s_lem in Word2Idx.keys()): 
                    token_ids.append(Word2Idx[s_lem])
                elif (v_lem in Word2Idx.keys()): 
                    token_ids.append(Word2Idx[v_lem])
                elif (a_lem in Word2Idx.keys()): 
                    token_ids.append(Word2Idx[a_lem])
                elif (n_lem in Word2Idx.keys()):
                    token_ids.append(Word2Idx[n_lem])
                else: 
                    token_ids.append(0)
            x.append(list(token_ids))
            # print(sum([1 if ele == 0 else 0 for ele in x[-1]])/len(x[-1]))
            # print(tokenizer.tokenize(line))
            # print(x[-1])
            # if (i > 30):
            #     exit(0)
            y.append(int(row[0]))
    return x, y

# def create_embedding_layer(embedding_matrix, trainable=False):
#     size, dim = embedding_matrix.shape
#     embedding_layer = nn.Embedding(size, dim)
#     embedding_layer.load_state_dict({'weight': Tensor(embedding_matrix)})
#     embedding_layer.weight.requires_grad = trainable
#     return embedding_layer