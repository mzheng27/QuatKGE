from torch import nn

class LSTM(nn.Module):

    # def __init__(self, embedding_matrix, maxpooling=False, conv_in_channels=300, conv_out_channels=1, kernel_size=5):
    #     super(CNN, self).__init__()
    #     self.embedding_layer = create_embedding_layer(embedding_matrix, trainable=False)
    #     self.conv = nn.Conv1d(conv_in_channels, conv_out_channels, kernel_size)
    #     if maxpooling:
    #         self.pooling = nn.MaxPool1d(conv_in_channels - kernel_size + 1)
    #     else:
    #         self.pooling = nn.AvgPool1d(conv_in_channels - kernel_size + 1)
    #     self.seq_layers = nn.Sequential(
    #         nn.ReLU(),
    #         nn.Linear(conv_out_channels, 1),
    #         nn.Sigmoid() 
    #     )
    def __init__(self, embedding_matrix, input_dim, hidden_dim, layer_dim, output_dim):
            super(LSTM, self).__init__()
            # self.embedding_layer = create_embedding_layer(embedding_matrix, trainable=False)

            self.hidden_dim = hidden_dim
            self.layer_dim = layer_dim

            self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.1)
            self.out_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # embedded = self.embedding_layer(x)
        # embedded = embedded.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x) 
        out = self.out_fc(hn[-1]) 
        # print(out.shape)
        return out
