import numpy as np
import torch
import os
import json

from torch.utils.data import Dataset
from torch.nn import functional as F

from datasets import load_dataset
from PIL import Image, ImageOps
import PIL
import pickle
            
class IAM_SROIE_Dataset(Dataset):
    def __init__(self, tokenizer, split="train"):
        super(IAM_SROIE_Dataset, self).__init__()
        self.data = load_dataset("priyank-m/iam_sroie_text_recognition", split=split)
        self.tokenizer = tokenizer
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        image = self.data[idx]['image']
        pixel_values = torch.tensor(self.preprocess(image)).permute(2, 0, 1).float()
        labels = self.tokenizer(self.data[idx]['text'], padding="max_length", max_length=64)
        return pixel_values, torch.tensor(labels.input_ids), torch.tensor(labels.attention_mask)
        
    
    def preprocess(self, image):
        w, h = image.size
        desired_size = max(w, h)
        delta_w = desired_size - w
        delta_h = desired_size - h
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        image = ImageOps.expand(image, padding)
        image = np.array(image.resize((224, 224), resample=PIL.Image.Resampling.BILINEAR))
        image = image / 255.0
        image = (image - self.mean) / self.std
        return image

def make_vocab_from_data():
    data = load_dataset("priyank-m/iam_sroie_text_recognition", split='train')
    all_chars = []
    for i in range(len(data)):
        a = [*data[i]['text']]
        all_chars += a
    vocab  = list(set(all_chars))
    vocab.append('[EOS]')
    vocab.append('[PAD]')
    vocab_dict = {}
    for i in range(len(vocab)):
        vocab_dict[vocab[i]] = i

    with open('vocab.pkl', 'wb') as handle:
        pickle.dump(vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    n = IAM_SROIE_Dataset()
    
    


    