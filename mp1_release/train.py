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
        for word_ids, label in data_loader:
            word_ids = word_ids.to(device)
            label = label.to(device)
            probs = model(word_ids)
            preds = torch.argmax(probs, 1)
            acc += torch.count_nonzero(preds == label)
                
        return acc/len(data_loader.dataset)
    

def get_val_loss(model, data_loader, device):
    model.eval()
    loss_list = []
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for word_ids, label in data_loader:
            word_ids = word_ids.to(device)
            label = label.to(device)
            pred_probs = model(word_ids)
            final_loss = loss(pred_probs, label)
            loss_list.append(final_loss.item())
    return loss_list

# np.set_printoptions(linewidth=np.inf)
def write_embed_file(model, vocab_dict, save_path, edim, device, fname="embed.txt"):
    with open(os.path.join(save_path, fname), 'w') as embed_file:
        embed_file.write(f'{len(vocab_dict)} {edim}\n')
        for key, value in vocab_dict.items():
            word_label = torch.tensor(value).unsqueeze(0).long().to(device)
            e = model.get_embeddings(word_label).squeeze().detach().cpu().numpy()
            line  = key + " " + np.array2string(e, max_line_width=np.inf)[1:-1] 
            line = ' '.join(line.split())
            embed_file.write(line+"\n")


if __name__ == "__main__":

    ## Load data
    vocab_dict = utils.get_word2ix(cfg.vocab_path)
    train_dataset = CBoWDataset(vocab_dict=vocab_dict, data_path=cfg.train_data_path, context_window=cfg.context_window)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
	)

    val_dataset = CBoWDataset(vocab_dict=vocab_dict, data_path=cfg.val_data_path, context_window=cfg.context_window)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
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
    model = CBoW2(vocab_size=len(train_dataset.vocab_dict), embedding_dim=cfg.edim)
    model.to(cfg.DEVICE)
    # model.apply(weight_init(module=nn.Linear, initf=nn.init.kaiming_uniform_))
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)
    optimizer.zero_grad()

    loss_ce = nn.CrossEntropyLoss()
    min_train_loss = math.inf
    min_val_loss = math.inf

    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        loss_list_train = []        
        
        for idx, (word_ids, label) in enumerate((tqdm(train_loader, desc="Batch: "))):
            optimizer.zero_grad()
            word_ids = word_ids.to(cfg.DEVICE)
            label = label.to(cfg.DEVICE)
            pred_probs = model(word_ids)
            final_loss = loss_ce(pred_probs, label)

            final_loss.backward()
            optimizer.step()

            loss_list_train.append(final_loss.item())
                    

        mean_train_loss = np.mean(np.array(loss_list_train))
        loss_list_val = get_val_loss(model=model, data_loader=val_loader, device=cfg.DEVICE)
        mean_val_loss = np.mean(np.array(loss_list_val))

        tqdm.write("Epoch - "+str(epoch+1)+" : Train CE Loss = "+str(mean_train_loss)+" | Val CE Loss = "+str(mean_val_loss))

        if min_val_loss >= mean_val_loss:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_val.torch"))
            tqdm.write("Current val best found at epoch="+str(epoch+1))
            min_val_loss = mean_val_loss 
            best_val_model = copy.deepcopy(model)

    tqdm.write("--------- Done Training ---------")

    tqdm.write("--------- Writing Embedding File ---------")
    write_embed_file(best_val_model, vocab_dict, save_imdt, cfg.edim, cfg.DEVICE, fname="val_embed.txt")

    tqdm.write("--------- Done ---------")


    
    
    
    

    