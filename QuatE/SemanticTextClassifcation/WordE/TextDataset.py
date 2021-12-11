from torch.utils.data import Dataset
from torch import Tensor
from embedding import create_embedding_layer

class TextDataset(Dataset):

    def __init__(self, X, y, embedding_matrix):
        self.X = X
        self.y = (Tensor(y) - 1).long()
        self.embedding_layer = create_embedding_layer(embedding_matrix, trainable=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.embedding_layer(Tensor(self.X[index]).long()), self.y[index]