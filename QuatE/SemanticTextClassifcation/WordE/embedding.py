import numpy as np
from torch import nn
from torch import Tensor
import csv
import torch

#load sentences as list of list of indexes
def load_data(Word2Idx, path):
    x = []
    y = []
    with open(path, 'r') as f: 
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        for i, row in enumerate(rdr): 
            line = ' '.join(row[1:]).lower()
            x.append(list(map(lambda x: Word2Idx[x] if x in Word2Idx.keys() else 0, line.split())))
            y.append(int(row[0]))
    return x, y

def create_embedding_layer(embedding_matrix, trainable=False):
    size, dim = embedding_matrix.shape
    embedding_layer = nn.Embedding(size, dim)
    embedding_layer.load_state_dict({'weight': Tensor(embedding_matrix)})
    embedding_layer.weight.requires_grad = trainable
    return embedding_layer