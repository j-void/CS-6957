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

def makeNGrams(files, w, vocab):
    ngrams_list = []
    out_list = []
    for file in files:
        with open(file) as f:
            lines = f.readlines()
        
        for line in lines:
            line_data = []
            for charac in line:
                if charac not in vocab.keys():
                    line_data.append("<unk>")
                else:
                    line_data.append(charac)
            
            for i in range(len(line_data)):
                if i < w:
                   r = w - i
                   l = ['[PAD]'] * r + line_data[:i]
                else:
                    l = line_data[i-w:i]
               
                ngrams_list.append(''.join(l))
                out_list.append(line_data[i])
            #     print(f'ngram={l}|out={line_data[i]}')
            # quit()
    
    return ngrams_list, out_list


def evaluateTest(files, w, vocab, count_dict):
    plist = []
    for file in files:
        with open(file) as f:
            lines = f.readlines()
        
        for line in lines:
            line_data = []
            probs = []
            ngrams_count = 0
            for charac in line:
                if charac not in vocab.keys():
                    line_data.append("<unk>")
                else:
                    line_data.append(charac)
            
            for i in range(len(line_data)):
                if i < w:
                   r = w - i
                   l = ['[PAD]'] * r + line_data[:i]
                else:
                    l = line_data[i-w:i]

                l = ''.join(l)

                try:
                    p = count_dict[l][line_data[i]]
                    t = sum(count_dict[l].values())
                except:
                    p = 0
                    t = 0
                
                probs.append((p+1)/(t+len(vocab.keys())))

            probs = np.array(probs)
            prob_mean = -np.mean(np.log2(probs))
            p = 2 ** prob_mean
            plist.append(p)
            
            
    
    return np.mean(np.array(plist))



def get_probability(ngrams_list, out_list, ngram, label):
    ngram_count = 0
    label_count = 0
    for i in range(len(ngrams_list)):
        if ngrams_list[i] == ngram:
            ngram_count += 1
            if out_list[i] == label:
                label_count += 1
    
    return (label_count+1)/(ngram_count+len(ngrams_list))

def get_prob_distribution(ngrams_list, out_list, vocab_dict):
    unique_ngrams = list(set(ngrams_list))
    count_dict = {k:0 for k in vocab_dict.keys()}
    prob_dict = {k:count_dict.copy() for k in unique_ngrams}
    for i in range(len(ngrams_list)):
        ngram = ngrams_list[i]
        out = out_list[i]
        prob_dict[ngram][out] += 1
    
    return prob_dict
    
    


if __name__ == "__main__":

    ## Loading data
    with  open(cfg.vocab_path, 'rb') as f:
        vocab_dict = pickle.load(f)
    
    train_data_files = sorted(utils.get_files(cfg.train_data_path))

    train_ngrams_list, train_out_list = makeNGrams(train_data_files, 4, vocab_dict)

    count_dict = get_prob_distribution(train_ngrams_list, train_out_list, vocab_dict)

    print(f"Total Count = {len(train_ngrams_list)}, Conditional Probabilities = {len(count_dict.keys())}")

    test_data_files = sorted(utils.get_files(cfg.test_data_path))

    test_preplexity = evaluateTest(test_data_files, 4, vocab_dict, count_dict)

    print(f"Test Preplexity = {test_preplexity}")




    




    
    
    
    

    