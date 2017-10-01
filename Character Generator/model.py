import torch
from torch import nn
from torch import autograd
import torch.optim as optim
import torch.nn.functional as F


class TextGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size, output_dim):
        super(TextGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = output_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers = num_layers, 
                            batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.scores = nn.Linear(hidden_dim, output_dim)
        self.batchnormD = nn.BatchNorm1d(hidden_dim)
        
    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.Tensor(self.num_layers,batch_size, self.hidden_dim).zero_()).cuda(),
                autograd.Variable(torch.Tensor(self.num_layers,batch_size, self.hidden_dim).zero_()).cuda())
    
    def forward(self, X, hidden):
        lstm_out, hidden = self.lstm(X.view(self.batch_size, X.size()[1], -1), hidden)
        lstm_dropout = self.dropout(lstm_out)
        batch = self.batchnormD(lstm_dropout.view(self.batch_size*X.size()[1], -1))
        tag_scores = F.log_softmax(self.scores(batch))
        tag_scores = tag_scores.view(self.batch_size,X.size()[1], -1)
        return tag_scores, hidden
    
    def forward2(self, X, hidden):
        lstm_out, hidden = self.lstm(X.view(1, X.size()[1], -1), hidden)
        lstm_dropout = self.dropout(lstm_out)
        batch = self.batchnormD(lstm_dropout.view(X.size()[1], -1))
        tag_scores = F.log_softmax(self.scores(batch))
        return tag_scores, hidden