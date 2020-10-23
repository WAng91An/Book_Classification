import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):

        # x: torch.Size([32, 1, 128, 300])
        # conv: Conv2d(1, 256, kernel_size=(2, 300), stride=(1, 1))
        # conv(x).shape: torch.Size([32, 256, 127, 1])

        x = F.relu(conv(x)).squeeze(3)
        # x: torch.Size([32, 256, 127])
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # torch.Size([32, 256, 1]) ->  torch.Size([32, 256])

        return x

    def forward(self, x):
        # x[0].shape: [batch_size, 128]   128: seq_length
        out = self.embedding(x[0]) # [batch_size, 128, 300]
        # out.shape: [batch_size, 128, 300]
        out = out.unsqueeze(1) # [batch_size, 1, 128, 300]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
