import torch
import math
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import os
import random
import torchtext
import scripts.state as state
import data_loader as dl

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

class TDParser2(nn.Module):

    def __init__(self, pose_set_dim, pos_embedding_dim, tagset_dim, glove_dim, combine='mean'):
        super(TDParser2, self).__init__()
        self.pose_set_dim = pose_set_dim
        self.tagset_dim = tagset_dim
        self.pos_embeddings = nn.Embedding(pose_set_dim, pos_embedding_dim)
        cdim = 2
        self.combine_type = combine
        if combine == 'mean':
            self.wlinear = nn.Linear(glove_dim, 200)
            self.plinear = nn.Linear(pos_embedding_dim, 200)
        elif combine == 'concatenate':
            self.wlinear = nn.Linear(glove_dim*cdim*2, 200)
            self.plinear = nn.Linear(pos_embedding_dim*cdim*2, 200)
        else:
            TypeError("Combine type not set")
        
        self.classifier = nn.Linear(200, tagset_dim)
        self.relu = nn.ReLU()


    def forward(self, word_embeds, pos_labels):
        pos_embeds = self.pos_embeddings(pos_labels)

        if self.combine_type=="mean":
            word_embeds = torch.mean(word_embeds, dim=1)
            pos_embeds = torch.mean(pos_embeds, dim=1)
        else:
            word_embeds = torch.cat([word_embeds[:,0,:], word_embeds[:,1,:], word_embeds[:,2,:] , word_embeds[:,3,:]], dim=1)
            pos_embeds = torch.cat([pos_embeds[:,0,:], pos_embeds[:,1,:], pos_embeds[:,2,:] , pos_embeds[:,3,:]], dim=1)

        out = self.relu(self.wlinear(word_embeds) + self.plinear(pos_embeds))
        out = self.classifier(out)
        
        return out
    
class TDParserCM(nn.Module):

    def __init__(self, pose_set_dim, pos_embedding_dim, tagset_dim, glove_dim, parse_set_dim, combine='mean'):
        super(TDParserCM, self).__init__()
        self.pose_set_dim = pose_set_dim
        self.tagset_dim = tagset_dim
        self.pos_embeddings = nn.Embedding(pose_set_dim, pos_embedding_dim)
        self.ps_embeddings = nn.Embedding(parse_set_dim, 50)
        cdim = 2
        self.combine_type = combine
        if combine == 'mean':
            self.wlinear = nn.Linear(glove_dim, 200)
            self.plinear = nn.Linear(pos_embedding_dim, 200)
            self.ps_linear = nn.Linear(50, 200)
        elif combine == 'concatenate':
            self.wlinear = nn.Linear(glove_dim*(cdim*2+4), 200)
            self.plinear = nn.Linear(pos_embedding_dim*(cdim*2+4), 200)
            self.ps_linear = nn.Linear(50*4, 200)
        else:
            TypeError("Combine type not set")
        
        self.classifier = nn.Linear(200, tagset_dim)
        self.relu = nn.ReLU()


    def forward(self, word_embeds, pos_labels, ps_labels):
        pos_embeds = self.pos_embeddings(pos_labels)
        ps_embeds = self.ps_embeddings(ps_labels)

        if self.combine_type=="mean":
            word_embeds = torch.mean(word_embeds, dim=1)
            pos_embeds = torch.mean(pos_embeds, dim=1)
            ps_embeds = torch.mean(ps_embeds, dim=1)
        else:
            word_embeds = word_embeds.flatten(1, 2)
            pos_embeds = pos_embeds.flatten(1, 2)
            ps_embeds = ps_embeds.flatten(1, 2)

        out = self.relu(self.wlinear(word_embeds) + self.plinear(pos_embeds)+ self.ps_linear(ps_embeds))
        out = self.classifier(out)
        
        return out
    

if __name__ == "__main__":
    pass

