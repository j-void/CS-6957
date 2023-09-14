import numpy as np
import torch
import os
import json
import scripts.utils as utils


from torch.utils.data import Dataset
from torch.nn import functional as F

class CBoWDataset(Dataset):
    def __init__(self, vocab_dict, data_path, context_window):
        super(CBoWDataset, self).__init__()
        self.vocab_dict = vocab_dict
        self.data_files = sorted(utils.get_files(data_path))
        self.data = utils.process_data(files=self.data_files, context_window=context_window, word2ix=self.vocab_dict)
        self.data_list = self.make_splits(self.data)
        self.context_window = context_window

    def __len__(self):
        return len(self.data_list)

    def make_splits(self, data):
        data_list = []
        for s in data:
            for i in range(len(s)-10):
                data_list.append(s[i:i+11])
        return data_list

    def __getitem__(self, idx):
        current_data_list = self.data_list[idx].copy()
        label = torch.tensor(current_data_list[self.context_window]).long()
        current_data_list.pop(self.context_window)
        word_ids = torch.tensor(current_data_list).long()
        return word_ids, label
            


if __name__ == "__main__":
    vocab_dict = utils.get_word2ix("vocab.txt")
    dataset = CBoWDataset(vocab_dict=vocab_dict, data_path="data/train", context_window=5)
    print(dataset.data_list[1])
    part = dataset.data_list[1] #dataset.data[0][1:12]
    inv_map = {v: k for k, v in vocab_dict.items()}
    words = [inv_map[x] for x in part]
    print(words[5])
    words.pop(5)
    print(words)


    