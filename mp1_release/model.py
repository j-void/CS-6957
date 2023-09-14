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

class CBoW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBoW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, inputs):
        # print(inputs.shape, self.embeddings(inputs).shape)
        embeds = torch.sum(self.embeddings(inputs), dim=1)
        # print(embeds.shape)
        out = self.classifier(embeds)
        #probs = F.softmax(out, dim=1)
        return out
    
    def get_embeddings(self, label):
        W = self.classifier.weight.clone()#.transpose(0, 1)
        one_hot = torch.nn.functional.one_hot(label, num_classes=self.vocab_size).float()
        output = torch.einsum('bv,ve->be', one_hot, W)
        return output
    
class CBoW2(nn.Module):

    def __init__(self, vocab_size, embedding_dim, device="cuda"):
        super(CBoW2, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        W = torch.randn((embedding_dim, vocab_size))
        self.register_parameter("W", torch.nn.Parameter(W))
        self.vocab_size = vocab_size

    def forward(self, inputs):
        embeds = torch.sum(self.embeddings(inputs), dim=1)
        out = torch.einsum('be,ev->bv', embeds, self.W)
        return out
    
    def get_embeddings(self, label):
        W = self.W.clone().transpose(0, 1)
        one_hot = torch.nn.functional.one_hot(label, num_classes=self.vocab_size).float()
        output = torch.einsum('bv,ve->be', one_hot, W)
        return output


if __name__ == "__main__":
    model = CBoW2(vocab_size=1000, embedding_dim=10, device="cpu")
    a = torch.tensor([23, 45, 78, 56]).unsqueeze(0)
    out = model(a)
    print(out.shape, torch.sum(out, dim=1))

