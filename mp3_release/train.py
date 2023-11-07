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

    

def get_preplexity(model, data_loader, device):
    model.eval()
    p_list = []
    loss_list = []
    loss = nn.CrossEntropyLoss(reduction='none', ignore_index=384)
    with torch.no_grad():
        for input_, target_ in data_loader:
            input_ = input_.to(cfg.DEVICE)
            target_ = target_.to(cfg.DEVICE)
            final_loss = model(input_, target_, loss)
            loss_list.append(torch.mean(final_loss).item())  
            final_loss = final_loss.view(input_.shape[0], -1)
            pad_counts = (input_ == 384).sum(dim=1)

            avg_counts = torch.zeros(pad_counts.shape)
            for b in range(final_loss.shape[0]):
                a = 500 - pad_counts[b]
                a = a + 1 if a < 499 else a
                avg_counts[b] = torch.mean(final_loss[b,:a+1])

            p = torch.exp(avg_counts)
            p_list.append(torch.mean(p).item())
                      
        
    return np.mean(np.array(p_list)), np.mean(np.array(loss_list))



if __name__ == "__main__":

    ## Load data
    with  open("data/vocab.pkl", 'rb') as f:
        vocab_dict = pickle.load(f)
        
    train_dataset = LSTMDataset(vocab_dict=vocab_dict, data_path=cfg.train_data_path, context_window=cfg.context_window)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
	)

    val_dataset = LSTMDataset(vocab_dict=vocab_dict, data_path=cfg.val_data_path, context_window=cfg.context_window)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
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
    model = LSTM_LanguageModel(vocab_size=len(vocab_dict), embedding_dim=cfg.edim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, context_window=cfg.context_window)
    model.to(cfg.DEVICE)
    model.apply(weight_init(module=nn.Linear, initf=nn.init.kaiming_normal_))
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)
    optimizer.zero_grad()

    num_param = sum(p.numel () for p in model.parameters ())
    tqdm.write("No. of Parameters = "+str(num_param))

    min_train_loss = math.inf
    min_val_loss = math.inf

    total_count = sum([v for v in train_dataset.count_dict.values()])
    weights = []
    for k,v in train_dataset.count_dict.items():
        w = 1 - (v/total_count)
        weights.append(w)

    loss_ce_weighted = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(cfg.DEVICE), ignore_index=384)   

    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        loss_list_train = []        
        
        for idx, (input_, target_) in enumerate((tqdm(train_loader, desc="Batch: "))):
            optimizer.zero_grad()
            input_ = input_.to(cfg.DEVICE)
            target_ = target_.to(cfg.DEVICE)
            final_loss = model(input_, target_, loss_ce_weighted)

            final_loss.backward()
            optimizer.step()

            loss_list_train.append(final_loss.item())
                    

        mean_train_ce_loss = np.mean(np.array(loss_list_train))
        val_preplexity, val_ce_loss = get_preplexity(model=model, data_loader=val_loader, device=cfg.DEVICE)

        tqdm.write("Epoch - "+str(epoch+1)+" : Train CE Loss = "+str(mean_train_ce_loss)+" | Val Preplexity = "+str(val_preplexity) + "| Val CE Loss = "+str(val_ce_loss))

        if min_val_loss >= val_preplexity:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_val.torch"))
            tqdm.write("Current val best found at epoch="+str(epoch+1))
            min_val_loss = val_preplexity 
            best_val_model = copy.deepcopy(model)

    tqdm.write("--------- Done Training ---------")


    test_dataset = LSTMDataset(vocab_dict=vocab_dict, data_path=cfg.test_data_path, context_window=cfg.context_window)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
	)

    test_preplexity, _ = get_preplexity(model=best_val_model, data_loader=test_loader, device=cfg.DEVICE)
    tqdm.write("Best Preplexity on Test Set = "+ str(test_preplexity))
    
    
    

    