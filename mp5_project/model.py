import torch
import math
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import os
import random
from transformers import VisionEncoderDecoderModel, ViTModel, ViTConfig


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

def weight_init(module, initf):
	def foo(m):
		classname = m.__class__.__name__.lower()
		if isinstance(m, module):
			initf(m.weight)
	return foo


class ViTEncoderLSTMDecoder(nn.Module):

    def __init__(self, vocab_size):
        super(ViTEncoderLSTMDecoder, self).__init__()
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.hidden_dim = 200
        self.decoder = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, vocab_size),
        ) 
        self.decoder.apply(weight_init(module=nn.Linear, initf=nn.init.kaiming_normal_))
        self.mlp.apply(weight_init(module=nn.Linear, initf=nn.init.kaiming_normal_))
        
        

    def forward(self, x):
        vout = self.encoder(x).last_hidden_state[:,0:1,:].expand(-1, 64, -1)
        lstm_out, (_, _) = self.decoder(vout)
        output = self.mlp(lstm_out.reshape(-1, self.hidden_dim))
        return output
	
    def generate(self, x):
        vout = self.encoder(x).last_hidden_state[:,0:1,:].expand(-1, 64, -1)
        lstm_out, (_, _) = self.decoder(vout)
        output = self.mlp(lstm_out.reshape(-1, self.hidden_dim))
        output_labels = torch.argmax(output, dim=1).view(-1, 64)
        return output_labels

if __name__ == "__main__":
    model = ViTEncoderLSTMDecoder(30000)
    a = torch.rand(1, 3, 224, 224)

    outputs = model(a)
    print(outputs.shape)
    


