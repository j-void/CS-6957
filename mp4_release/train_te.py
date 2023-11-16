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

def weight_init(module, initf):
	def foo(m):
		classname = m.__class__.__name__.lower()
		if isinstance(m, module):
			initf(m.weight)
	return foo


def get_accuracy(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        acc = 0.0
        for ids, mask, token_type_ids, target in data_loader:
            ids = ids.to(cfg.DEVICE)
            mask = mask.to(cfg.DEVICE)
            token_type_ids = token_type_ids.to(cfg.DEVICE)
            target = target.to(cfg.DEVICE)
            probs = model(ids, mask, token_type_ids)
            probs = torch.softmax(probs, dim=1)
            preds = torch.argmax(probs, 1)
            acc += torch.count_nonzero(preds == target).item()
                
        return acc/len(data_loader.dataset)

def get_random_accuracy(dataset):

    with torch.no_grad():
        acc = 0.0
        target_labels = [0, 1]
        for gt in dataset.data['label']:
            pred = random.choice(target_labels)
            acc += (pred == gt)
                
        return acc/len(dataset.data['label'])

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

    if cfg.model == "tiny":
        print("--------- Using BERT-Tiny ---------")
        tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    else:
        print("--------- Using BERT-Mini ---------")
        tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
	
    train_dataset = RTEDataset(tokenizer=tokenizer, max_length=cfg.max_length, split='train')
    val_dataset = RTEDataset(tokenizer=tokenizer, max_length=cfg.max_length, split='validation')
    test_dataset = RTEDataset(tokenizer=tokenizer, max_length=cfg.max_length, split='test')
	
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
	)

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
	)

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
	)

    save_imdt = os.path.join(cfg.checkpoint_dir, "output")
    if not os.path.exists(save_imdt):
        os.makedirs(save_imdt)
    
    save_dir = os.path.join(cfg.checkpoint_dir, "save")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("--------- Initializing Neworks ---------")
    if cfg.model == "tiny":
        model = BERT_Tiny(finetune=cfg.finetune)
    else:
        model = BERT_Mini(finetune=cfg.finetune)

    model.to(cfg.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), cfg.learning_rate)
    optimizer.zero_grad()

    
    min_train_loss = math.inf
    max_val_acc = 0

    loss_ce = nn.CrossEntropyLoss()   

    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        loss_list_train = []        
        
        for idx, (ids, mask, token_type_ids, target) in enumerate((tqdm(train_loader, desc="Batch: "))):
            optimizer.zero_grad()
            ids = ids.to(cfg.DEVICE)
            mask = mask.to(cfg.DEVICE)
            token_type_ids = token_type_ids.to(cfg.DEVICE)
            target = target.to(cfg.DEVICE)
            output = model(ids, mask, token_type_ids)

            final_loss = loss_ce(output, target)
            final_loss.backward()
            optimizer.step()

            loss_list_train.append(final_loss.item())
                    

        mean_train_loss = np.mean(np.array(loss_list_train))
        val_acc = get_accuracy(model=model, data_loader=val_loader, device=cfg.DEVICE)

        tqdm.write("Epoch - "+str(epoch+1)+" : Train CE Loss = "+str(mean_train_loss)+" | Val Accuracy = "+str(val_acc))

        if max_val_acc <= val_acc:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_val.torch"))
            tqdm.write("Current val best found at epoch="+str(epoch+1))
            max_val_acc = val_acc 
            best_val_model = copy.deepcopy(model)

    tqdm.write("--------- Done Training ---------")


    test_acc= get_accuracy(model=best_val_model, data_loader=test_loader, device=cfg.DEVICE)
    random_acc = get_random_accuracy(dataset=test_dataset)
    tqdm.write("Best Accuracy on Test Set = "+ str(test_acc)+" & Random Baseline Accuracy = "+str(random_acc))

    tqdm.write("Run on Hidden Set")
    run_hidden(best_val_model, tokenizer, os.path.join(save_imdt, 'hidden_rte.csv'))

    tqdm.write("Done")
    
    
    

    