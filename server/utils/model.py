import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, vocab_size, output_dim):
        super().__init__()
        self.linear = nn.Linear(vocab_size, output_dim)
    def forward(self, x):
        return self.linear(x)

class BasicDNN(nn.Module):
    def __init__(self, vocab_size, output_dim):
        super().__init__()
        hidden_dim = 100
        self.linear1 = nn.Linear(vocab_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                    dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels = 1, 
            out_channels = n_filters, 
            kernel_size = (fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = self.embedding(text)
        text = text.unsqueeze(1)
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        conved = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(conved, dim = 1))
        return F.softmax(self.fc(cat),1)
