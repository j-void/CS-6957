import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict
import math
import config as cfg
from data_loader import *
from model import *
import copy
from transformers import BertTokenizer
import pandas as pd

## set seeds
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(134)


def run_hidden(model, tokenizer, save_path):
    df = pd.read_csv("hidden_rte.csv")
    text1 = df['text1'].values.tolist()
    text2 = df['text2'].values.tolist()
    prediction = []
    probab_0 = []
    probab_1 = []

    tokenized_inp = tokenizer(text1, text2, truncation=True ,padding=True ,return_tensors ='pt', max_length=cfg.max_length)

    model.eval()

    with torch.no_grad():
        for idx in range(len(text1)):
            ids = tokenized_inp['input_ids'][idx].long().unsqueeze(0).to(cfg.DEVICE)
            mask = tokenized_inp['attention_mask'][idx].long().unsqueeze(0).to(cfg.DEVICE)
            token_type_ids = tokenized_inp["token_type_ids"][idx].long().unsqueeze(0).to(cfg.DEVICE)
            probs = model(ids, mask, token_type_ids)
            pred = torch.argmax(probs, 1)
            prediction.append(pred.squeeze().item())
            probs = torch.softmax(probs, dim=1)
            probab_0.append(round(probs[:,0].squeeze().item(), 3))
            probab_1.append(round(probs[:,1].squeeze().item(), 3))

    
    df['prediction'] = prediction
    df['probab_0'] = probab_0
    df['probab_1'] = probab_1

    df.to_csv(save_path, index=False)







if __name__ == "__main__":

    ## Best on Bert-mini
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
	
    print("----Running on RTE----")
    model = BERT_Mini(finetune=cfg.finetune)
    model.load_state_dict(torch.load("checkpoints/rte_ft_mini_lr_0_0001/save/model_val.torch"))
    model.to(cfg.DEVICE)
    
    text1 = ["The doctor is prescribing medicine.", "The doctor is prescribing medicine.", "The nurse is tending to the patient.", "The nurse is tending to the patient."]
    text2 = ["She is prescribing medicine.", "He is prescribing medicine.", "She is tending to the patient.", "He is tending to the patient."]
    
    tokenized_inp_rte = tokenizer(text1, text2, truncation=True ,padding=True ,return_tensors ='pt', max_length=cfg.max_length)
    
    model.eval()
    with torch.no_grad():
        for idx in range(tokenized_inp_rte['input_ids'].shape[0]):
            ids = tokenized_inp_rte['input_ids'][idx].long().unsqueeze(0).to(cfg.DEVICE)
            mask = tokenized_inp_rte['attention_mask'][idx].long().unsqueeze(0).to(cfg.DEVICE)
            token_type_ids = tokenized_inp_rte["token_type_ids"][idx].long().unsqueeze(0).to(cfg.DEVICE)
            output = model(ids, mask, token_type_ids)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, 1).squeeze().item()
            label = "entailment" if pred == 0 else "not entailment"
            print(f"P: {text1[idx]}; H: {text2[idx]}; Lable: {label}")
        
    print("----Running on SST----")
    model = BERT_Mini(finetune=cfg.finetune)
    model.load_state_dict(torch.load("checkpoints/sst_ft_mini_lr_0_0001/save/model_val.torch"))
    model.to(cfg.DEVICE)
    
    text = ["Kate should get promoted, she is an amazing employee.", "Bob should get promoted, he is an amazing employee.", \
        "Kate should get promoted, he is an amazing employee.", "Bob should get promoted, they are an amazing employee."]
    
    tokenized_inp_sst = tokenizer(text, truncation=True ,padding=True ,return_tensors ='pt', max_length=cfg.max_length)
    
    model.eval()
    with torch.no_grad():
        for idx in range(tokenized_inp_sst['input_ids'].shape[0]):
            ids = tokenized_inp_sst['input_ids'][idx].long().unsqueeze(0).to(cfg.DEVICE)
            mask = tokenized_inp_sst['attention_mask'][idx].long().unsqueeze(0).to(cfg.DEVICE)
            token_type_ids = tokenized_inp_sst["token_type_ids"][idx].long().unsqueeze(0).to(cfg.DEVICE)
            output = model(ids, mask, token_type_ids)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, 1)
            label = "positive" if pred == 0 else "negative"
            print(f"S: {text[idx]}; Lable: {label}")
        
        
    
    
    
    

    