import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict
import math
import scripts.utils as utils
import config as cfg
from data_loader import *
from model import *
import copy

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(1)




if __name__ == "__main__":

    ## Load data
    with  open("data/vocab.pkl", 'rb') as f:
        vocab_dict = pickle.load(f)
    inv_vocab = {v: k for k, v in vocab_dict.items()}
    ## config variables
    save_dir = ""
    edim = 50
    hidden_dim = 200
    num_layers = 1
    context_window = 500
    DEVICE = 'cuda'
    
    examples = ["The little boy was", "Once upon a time in", "With the target in", "Capitals are big cities. For example,", "A cheap alternative to"]

    model = LSTM_LanguageModel(vocab_size=len(vocab_dict), embedding_dim=edim, hidden_dim=hidden_dim, num_layers=num_layers, context_window=context_window)
    model.load_state_dict(torch.load("checkpoints/n1_lr_0_0001/save/model_val.torch"))
    model.to(DEVICE)
    model.eval()
    
    complete_sentences = []
    
    with torch.no_grad():     
        for ex in examples:
            sentence = ex
            input_ = torch.tensor([vocab_dict[c] for c in ex]).unsqueeze(0).to(DEVICE)
            for k in range(200):
                output = model.generate(input_)
                prob = output.exp()
                word_id = torch.multinomial(prob, num_samples=1)
                
                input_ = torch.cat((input_, word_id[-1].unsqueeze(0)), dim=1)

                new_char = inv_vocab[word_id[-1].item()]
                sentence += new_char
            complete_sentences.append(sentence)
    
    with open('generated.txt', 'w') as f:
        for i, line in enumerate(complete_sentences):
            f.write(f"{i+1} : {line}\n")
    
    print("Written to generated.txt")
            
                

                
            
    
    
    

    