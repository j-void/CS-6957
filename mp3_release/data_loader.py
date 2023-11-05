import numpy as np
import torch
import os
import json
import scripts.utils as utils
import  pickle

from torch.utils.data import Dataset
from torch.nn import functional as F

class LSTMDataset(Dataset):
    def __init__(self, vocab_dict, data_path, context_window, count_dict=None):
        super(LSTMDataset, self).__init__()
        self.vocab_dict = vocab_dict
        self.data_files = sorted(utils.get_files(data_path))
        self.data = utils.convert_files2idx(self.data_files, self.vocab_dict)

        self.context_window = context_window
        self.data_list, self.target_list, count_dict_ = self.make_splits(self.data, context_window)
        if count_dict == None:
            self.count_dict = count_dict_
        else:
            self.count_dict = count_dict

    def __len__(self):
        return len(self.data_list)

    def make_splits(self, data, cw):
        data_list = []
        target_list = []
        count_dict = {k:0 for k in self.vocab_dict.values()}
        for s in data:
            for i in range(0, len(s), cw):
                data = s[i:i+cw]
                if len(data) < cw:
                    data += [self.vocab_dict['[PAD]']]*abs(cw-len(data))
                target = s[i+1:i+1+cw]
                if len(target) < cw:
                    target += [self.vocab_dict['[PAD]']]*abs(cw-len(target))
                data_list.append(data)
                target_list.append(target)
                for d in data:
                    count_dict[d] += 1

                
        return data_list, target_list, count_dict

    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx]).long(), torch.tensor(self.target_list[idx]).long()
            


if __name__ == "__main__":
    
    with  open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    
    n = LSTMDataset(vocab, "data/train", 500)
    print(n[399])
    


    