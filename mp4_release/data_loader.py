import numpy as np
import torch
import os
import json
import  pickle

from torch.utils.data import Dataset
from torch.nn import functional as F


from datasets import load_dataset


class RTEDataset(Dataset):
    def __init__(self, tokenizer, max_length, split="train"):
        super(RTEDataset, self).__init__()
        self.data = load_dataset("yangwang825/rte", split=split)
        self.tokenized_inp = tokenizer(self.data['text1'], self.data['text2'], truncation=True ,padding=True ,return_tensors ='pt', max_length=max_length)

    def __len__(self):
        return self.tokenized_inp['input_ids'].shape[0]

    def __getitem__(self, idx):
        ids = self.tokenized_inp['input_ids'][idx].long()
        mask = self.tokenized_inp['attention_mask'][idx].long()
        token_type_ids = self.tokenized_inp["token_type_ids"][idx].long()
        target = torch.tensor(self.data['label'][idx]).long()
        return ids, mask, token_type_ids, target
    

class SSTDataset(Dataset):
    def __init__(self, tokenizer, max_length, split="train"):
        super(SSTDataset, self).__init__()
        self.data = load_dataset("gpt3mix/sst2", split=split)
        self.tokenized_inp = tokenizer(self.data['text'], truncation=True, padding=True, return_tensors ='pt', max_length=max_length)

    def __len__(self):
        return self.tokenized_inp['input_ids'].shape[0]

    def __getitem__(self, idx):
        ids = self.tokenized_inp['input_ids'][idx].long()
        mask = self.tokenized_inp['attention_mask'][idx].long()
        token_type_ids = self.tokenized_inp["token_type_ids"][idx].long()
        target = torch.tensor(self.data['label'][idx]).long()
        return ids, mask, token_type_ids, target
            


if __name__ == "__main__":
    
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    n = SSTDataset(tokenizer=tokenizer, max_length=512)
    print(n.tokenized_inp['input_ids'].shape)
    


    