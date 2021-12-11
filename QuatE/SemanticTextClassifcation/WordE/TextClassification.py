import pickle
import bcolz
import os
import torch
import sys

from cnn import LSTM
from embedding import load_data
from TextDataset import TextDataset
from torch.utils.data import DataLoader
from torch.optim import Adagrad
from torch.nn.functional import binary_cross_entropy
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence


""""input to collate_fn is a batch of data with the batch size in DataLoader, 
pad each batch"""
def collate_fn(batch): 
    (xx, yy) = zip(*batch)
    x_lens = [x.shape[0] for x in xx]
    xx_pad = pad_sequence(xx, padding_value=0)
    x_packed = pack_padded_sequence(xx_pad, x_lens, enforce_sorted=False)
    return x_packed, torch.Tensor(yy).long()

def train(device, train_dataloader, val_dataloader, optimizer, loss_f, epoch, model, log_interval=50):
    model.train()
    for i in range(epoch):
        pid = os.getpid()
        for batch_i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            y = y.reshape((-1,))
            optimizer.zero_grad()
            output = model(X)
            # print(y)
            # print(output)
            loss = loss_f(output, y)
            loss.backward()
            optimizer.step()
            if batch_i % log_interval == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid, i+1, batch_i * len(X), len(train_dataloader.dataset),
                                100. * batch_i / len(train_dataloader), loss.item()))
        print("\nTrain Epoch {} test".format(i+1))
        test(device, val_dataloader, nn.CrossEntropyLoss(reduction="sum"), model)
        model.train()
        print("\n")

def test(device, dataloader, loss_f, model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model.forward(X)
            test_loss += loss_f(output, y).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(y).sum().item()
    test_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    print('average loss: {:.4f}, accuracy: {}/{} ({:.4f}%)'.format(test_loss, correct, len(dataloader.dataset), 100. * accuracy))

def main():
    GLOVE_PATH = "./data"
    EMBEDDING_MATRIX = bcolz.open(f'{GLOVE_PATH}/840B.300.dat')[:]
    #word index in matrix 
    WORD2INDEX = pickle.load(open(f'{GLOVE_PATH}/840B.300_idx.pkl', 'rb'))

    print("start loading data")
    test_path = 'data/ag_news_csv/test.csv'
    train_path = 'data/ag_news_csv/train.csv'
    train_X, train_y = load_data(WORD2INDEX, train_path)
    # val_X, val_y = load_data(WORD2INDEX, val_path)
    test_X, test_y = load_data(WORD2INDEX, test_path)
    train_set = DataLoader(TextDataset(train_X, train_y, EMBEDDING_MATRIX), batch_size=64, shuffle=True, collate_fn=collate_fn)
    # val_set = DataLoader(TextDataset(val_X, val_y), batch_size=1024)
    test_set = DataLoader(TextDataset(test_X, test_y, EMBEDDING_MATRIX), batch_size=1024, shuffle=True, collate_fn=collate_fn)
    
    print("data loaded")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model_maxpooling = CNN(device, EMBEDDING_MATRIX, maxpooling=True, kernel_size=kernel_size)
    # model = CNN(EMBEDDING_MATRIX, maxpooling=maxpoooling, kernel_size=kernel_size, conv_out_channels=1).to(device)
    model = LSTM(EMBEDDING_MATRIX, input_dim=300, hidden_dim=20, layer_dim=1, output_dim=4)
    model = model.to(device)
    optimizer = Adagrad(model.parameters(), lr=1e-3, lr_decay=3e-5)
    # loss_f = binary_cross_entropy
    loss_f = nn.CrossEntropyLoss()
    epoch = 300

    print("start training")
    train(device, train_set, test_set, optimizer, loss_f, epoch, model, log_interval=100)
    print("training completed")
    # print("final validation")
    # test(device, val_set, loss_f, model)
    # print("validation completed")
    print("test:")
    test(device, test_set, loss_f, model)
    print("test completed")

    #modify the path to save different models
    #To load the parameters, use:
    # model.load_state_dict(torch.load('PATH'))
    # save_path = "./param/" + pooling_scheme + "_kernelsize_" + str(kernel_size) + ".pth"
    # torch.save(model.state_dict(), save_path)
    # print("\nmodel saved at", save_path)

if __name__ == "__main__":
    main()
