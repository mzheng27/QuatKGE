from torch.utils.data import Dataset
from torch import Tensor
import torch
# from embedding import create_embedding_layer

class TextDataset(Dataset):

    def __init__(self, X, y, embedding_matrix):
        self.X = X
        self.y = (Tensor(y) - 1).long()
        self.embedding_matrix = embedding_matrix
       
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        #X is a list of list 
        #X[i] is a list of index of the ith sentence
        batch_embedding = []
        for i in range(len(self.X[index])): 
            word2list = self.embedding_matrix['emb_x_a.weight'][self.X[index][i]] + \
            self.embedding_matrix['emb_y_a.weight'][self.X[index][i]] + \
            self.embedding_matrix['emb_z_a.weight'][self.X[index][i]] + \
            self.embedding_matrix['emb_s_a.weight'][self.X[index][i]]
            batch_embedding.append(word2list)

        return Tensor(batch_embedding), self.y[index]
