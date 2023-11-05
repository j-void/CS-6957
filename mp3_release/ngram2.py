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

def makeNGrams(data, w, vocab):
    ngrams_list = []
    out_list = []
    inv_vocab = {v: k for k, v in vocab.items()}
    for s in data:
        for i in range(len(s)):
            if i < w:
                r = w - i
                l = [vocab['[PAD]']] * r + s[:i]
            else:
                l = s[i-w:i]
            lv = ''.join([inv_vocab[j] for j in l])
            ngrams_list.append(lv)
            out_list.append(inv_vocab[s[i]])
    
    return ngrams_list, out_list


def evaluateTest(data, w, vocab, count_dict):
    plist = []
    
    inv_vocab = {v: k for k, v in vocab.items()}
    
    for s in data:
        probs = []
        for i in range(len(s)):
            if i < w:
                r = w - i
                l = [vocab['[PAD]']] * r + s[:i]
            else:
                l = s[i-w:i]
                
            lv = ''.join([inv_vocab[j] for j in l])
            
            try:
                p = count_dict[lv][inv_vocab[s[i]]]
                t = sum(count_dict[lv].values())
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
    train_data = utils.convert_files2idx(train_data_files, vocab_dict)
    
    train_ngrams_list, train_out_list = makeNGrams(train_data, 4, vocab_dict)
    
    count_dict = get_prob_distribution(train_ngrams_list, train_out_list, vocab_dict)
    
    print(f"Total Count = {len(train_ngrams_list)}, Conditional Probabilities = {len(count_dict.keys())}")
    

    test_data_files = sorted(utils.get_files(cfg.test_data_path))
    test_data = utils.convert_files2idx(test_data_files, vocab_dict)

    test_preplexity = evaluateTest(test_data, 4, vocab_dict, count_dict)

    print(f"Test Preplexity = {test_preplexity}")




    




    
    
    
    

    