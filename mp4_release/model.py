import torch
import math
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import os
import random
from transformers import BertModel


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

class BERT_Tiny(nn.Module):

    def __init__(self, finetune=True):
        super(BERT_Tiny, self).__init__()

        self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny')
        if finetune==False:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.mlp = nn.Linear(128, 2) 
   

    def forward(self, ids, mask, token_type_ids):
        bert_output = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=True)
        output = self.mlp(bert_output['pooler_output'])
        return output 
    
class BERT_Mini(nn.Module):

    def __init__(self, finetune=True):
        super(BERT_Mini, self).__init__()

        self.bert = BertModel.from_pretrained('prajjwal1/bert-mini')
        if finetune==False:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.mlp = nn.Linear(256, 2)  
   

    def forward(self, ids, mask, token_type_ids):
        bert_output = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=True)
        output = self.mlp(bert_output['pooler_output'])
        return output 
    

    
    
    


if __name__ == "__main__":
    model = LSTM_LanguageModel(vocab_size=386, embedding_dim=50, hidden_dim=200, num_layers=1, context_window=500)
    a = torch.randint(0, 385, (8, 500))
    t = torch.randint(0, 385, (8, 500))
    loss = nn.CrossEntropyLoss()
    output = model(a, t, loss)
    print(output)

