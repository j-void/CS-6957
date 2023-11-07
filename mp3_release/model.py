import torch
import math
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import os
import random

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

class LSTM_LanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_window, num_layers, hidden_dim):
        super(LSTM_LanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnns = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        ) 
   

    def forward(self, inputs, targets, loss):
        # print(inputs.shape)
        embeds = self.embeddings(inputs)
        # print(embeds.shape)
        lstm_out, (_, _) = self.rnns(embeds)
        final_loss = 0

        # ## loop over batch - faster
        # for i in range(inputs.shape[0]):
        #     pred = self.mlp(lstm_out[i,:,:])
        #     final_loss += loss(pred, targets[i,:])

        # return final_loss/inputs.shape[0]
        # print(lstm_out.shape)
        output = self.mlp(lstm_out.reshape(-1, self.hidden_dim))
        final_loss = loss(output, targets.view(-1))

        return final_loss
    
    def generate(self, inputs):
        embeds = self.embeddings(inputs)
        lstm_out, (_, _) = self.rnns(embeds)
        output = self.mlp(lstm_out.reshape(-1, self.hidden_dim))
        return output
    
    
    


if __name__ == "__main__":
    model = LSTM_LanguageModel(vocab_size=386, embedding_dim=50, hidden_dim=200, num_layers=1, context_window=500)
    a = torch.randint(0, 385, (8, 500))
    t = torch.randint(0, 385, (8, 500))
    loss = nn.CrossEntropyLoss()
    output = model(a, t, loss)
    print(output)

