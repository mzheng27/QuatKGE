import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import json
import csv
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

#Reference: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#https://github.com/srviest/char-cnn-text-classification-pytorch
class AGNEWs(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0=1014):
        """Create AG's News dataset object.
        Arguments:
            label_data_path: The path of label and data file in csv.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        self.loadAlphabet(alphabet_path)
        self.load(label_data_path)
        self.l0=l0
            
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y

    def loadAlphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase = True):
        self.label = []
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                #label is the first letter of the sentence 
                self.label.append(int(row[0]))
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()                
                self.data.append(txt)

        self.y = torch.LongTensor(self.label)

    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        sequence = self.data[idx]
        # X = torch.zeros(len(self.alphabet), len(sequence))
        X = torch.zeros(len(sequence), len(self.alphabet))
        # X = torch.zeros(len(self.alphabet), self.l0)
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char)!=-1:
                X[index_char][self.char2Index(char)] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def getClassWeight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples/float(self.label.count(c)) for c in label_set]    
        return class_weight, num_class


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.layer_dim = layer_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.1)

            # Readout laye
            # self.out_fc = nn.Sequential(
            #     nn.Linear(hidden_dim, 54), 
            #     nn.ReLU(),
            #     nn.BatchNorm1d(64),
            #     nn.Dropout(0.1), 
            #     nn.Linear(64, 64), 
            #     nn.ReLU(),
            #     nn.BatchNorm1d(64),
            #     nn.Linear(64, output_dim), 
            #     nn.ReLU(),                
            #     )
            self.out_fc = nn.Linear(hidden_dim, output_dim)
            # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x = x.cuda()
        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).cuda()
        out, (hn, cn) = self.lstm(x) 
        out = self.out_fc(hn[-1]) 
        return out
        
def save_checkpoint(model, state, filename):
    model = model.module 
    state['state_dict'] = model.state_dict()
    torch.save(state,filename)

""""input to collate_fn is a batch of data with the batch size in DataLoader, 
pad each batch"""
def collate_fn(batch): 
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    # x_lens = torch.LongTensor([len(x) for x in xx])
    xx_pad = pad_sequence(xx, padding_value=0)
    x_packed = pack_padded_sequence(xx_pad, x_lens, enforce_sorted=False)
    # print(yy)
    return x_packed, (torch.Tensor(yy) - 1).long()


def train(model, optimizer, loss_fn, train_dl, epochs=100, device='cuda'):    
    model = model.to(device)
    model.train()
    for epoch in range(epochs): 
        for i_batch, data in enumerate(train_dl):
            x, y = data 
            y = y.reshape((-1,))

            # print("x shape", x.shape)
            x, y= x.to(device), y.to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 400)
            optimizer.step()
            
            corrects = (torch.max(yhat, 1)[1].view(y.size()).data == y.data).sum()

        train_acc = 100.0 * corrects / y.shape[0]
        # train_loss  = train_loss / len(train_dl.dataset)
        print('Epoch[{}] - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% ({}/{})'.format(epoch,
                                                                             loss.data,
                                                                             optimizer.state_dict()['param_groups'][0]['lr'],
                                                                             train_acc,
                                                                             corrects,
                                                                              y.shape[0]))
        

def test(model, loss_fn, test_dl, device='cuda'): 
    model.eval()
    num_test_correct = 0
    num_test_examples = 0
    for i_batch, data in enumerate(test_dl):
        x, y = data
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        
        num_test_correct  += (torch.max(yhat, 1)[1].view(y.size()).data == y.data).sum()
        num_test_examples += y.shape[0]

    test_acc  = num_test_correct / num_test_examples
    print('\rEvaluation - acc: {:.3f} '.format(test_acc))

if __name__ == '__main__':
    test_path = 'data/ag_news_csv/test.csv'
    train_path = 'data/ag_news_csv/train.csv'
    val_path = 'data/ag_news_csv/test.csv'
    alphabet_path = 'alphabet.json'

    train_dataset = AGNEWs(train_path, alphabet_path)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, drop_last=False, collate_fn=collate_fn, shuffle=True)
  
    # val_dataset = AGNEWs(val_path, alphabet_path)
    # val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, drop_last=True, collate_fn=collate_fn, shuffle=Ture)
  
    test_dataset = AGNEWs(test_path, alphabet_path)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, drop_last=False, collate_fn=collate_fn, shuffle=True)
    

    #Define parameters and create model 
    num_features = len(train_dataset.alphabet)
    l0 = 1014
    hidden_dim = 20 
    layer_dim = 1
    output_dim = 4
    model = LSTMModel(num_features, hidden_dim, layer_dim, output_dim)

    #Hyperparam
    num_epoch = 70
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Train & Test
    train(model, optimizer, criterion, train_loader, num_epoch, device=device) 
    test(model, criterion, test_loader, device=device)
