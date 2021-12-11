import numpy as np
from torch import nn
from torch import Tensor
import csv
import torch
def sentence2vector(Word2Idx, sentence, length):
    #map a sentence to a numpy array of word ids
    v = np.zeros((length,))
    words = sentence.split()
    for i in range(length):
        if i < len(words):
            try:
                v[i] = Word2Idx[words[i]]
            except:
                continue
        else:
            v[i] = 0
    return v 

def test_sentence2index():
    word_dict = {"hello": 0, "world": 1}
    sentence = "hello world"
    length = 2
    if sentence2vector(word_dict, sentence, length)[0] == 0 and sentence2vector(word_dict, sentence, length)[1] == 1:
        print("passed")
        return
    print("failed")

def load_data(Word2Idx, path, with_label=True, length=None):
    num_examples = 0
    max_len = 0
    with open(path, 'r') as f: 
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        for index, row in enumerate(rdr):     
            num_examples += 1
            line = ' '.join(row[1:]) 
            if (len(line.split(' ')) > max_len): 
                max_len = len(line.split(' '))
        if length == None: 
            length = max_len 
        input_dim = length 
        x = np.zeros((num_examples, input_dim))
        y = np.zeros((num_examples, ))
        with open(path, 'r') as f: 
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for i, row in enumerate(rdr): 
                line = ' '.join(row[1:]) 
                # print(line)
                if with_label:
                    x[i, :] = sentence2vector(Word2Idx, line, length)
                    # y[i] = float(row[0])
                    y[i] = int(row[0])
                else: 
                    x[i, :] = sentence2vector(Word2Idx, line, length)
    
    return x, torch.LongTensor(y)

def create_embedding_layer(embedding_matrix, trainable=False):
    size, dim = embedding_matrix.shape
    embedding_layer = nn.Embedding(size, dim)
    embedding_layer.load_state_dict({'weight': Tensor(embedding_matrix)})
    embedding_layer.weight.requires_grad = trainable
    return embedding_layer