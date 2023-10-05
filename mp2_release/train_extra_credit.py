import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict
import math
import scripts.state as state
import config as cfg
from data_loader import *
from model import *
import copy
from scripts.evaluate import compute_metrics
import eval_utils as util

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



if __name__ == "__main__":

    ## Load data
    pose_set = convert_to_labels(get_file_contents(cfg.pose_set_file))
    tagset = convert_to_labels(get_file_contents(cfg.tagset_file))
    reverse_tagset = dict(zip(tagset.values(), tagset.keys()))
    parse_set = convert_to_labels(get_parse_labels(cfg.tagset_file))
    
    glove = torchtext.vocab.GloVe(name=cfg.glove_name, dim=cfg.glove_dim)

    train_dataset = ParsingDataset2(data_file=cfg.train_data_path, pose_set=pose_set, tagset=tagset, glove=glove, \
        parse_set=parse_set, split='train')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
	)

    val_dataset = ParsingDatasetEval(data_file=cfg.val_data_path, pose_set=pose_set, tagset=tagset, glove=glove, split='val')

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available()
	)
    
    test_dataset = ParsingDatasetEval(data_file=cfg.test_data_path, pose_set=pose_set, tagset=tagset, glove=glove, split='test')

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available()
	)

    save_imdt = os.path.join(cfg.checkpoint_dir, "output")
    if not os.path.exists(save_imdt):
        os.makedirs(save_imdt)
    
    save_dir = os.path.join(cfg.checkpoint_dir, "save")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("--------- Initializing Networks ---------")

    model = TDParserCM(pose_set_dim=len(pose_set), pos_embedding_dim=cfg.pos_embedding_dim, \
        glove_dim=cfg.glove_dim, tagset_dim=len(tagset), parse_set_dim =len(parse_set), combine=cfg.combine_type)
    model.to(cfg.DEVICE)
    model.apply(weight_init(module=nn.Linear, initf=nn.init.xavier_normal_))
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate, weight_decay=1e-3)
    optimizer.zero_grad()

    loss_ce = nn.CrossEntropyLoss()
    val_acc = 0


    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        loss_list_train = []        
        
        for idx, (word_embeds, pos_labels, parse_labels, action_label) in enumerate((tqdm(train_loader, desc="Batch: "))):
            optimizer.zero_grad()
            word_embeds = word_embeds.to(cfg.DEVICE)
            pos_labels = pos_labels.to(cfg.DEVICE)
            parse_labels = parse_labels.to(cfg.DEVICE)
            action_label = action_label.to(cfg.DEVICE)
            
            pred_action_probs = model(word_embeds, pos_labels, parse_labels)
            final_loss = loss_ce(pred_action_probs, action_label)

            final_loss.backward()
            optimizer.step()

            loss_list_train.append(final_loss.item())
                    

        mean_train_loss = np.mean(np.array(loss_list_train))


        current_val_acc, _ = util.get_accuracy_cm(model=model, data_loader=val_loader, parse_set=parse_set, \
            reverse_tagset=reverse_tagset, device=cfg.DEVICE)

        tqdm.write("Epoch - "+str(epoch+1)+" : Train CE Loss = "+str(mean_train_loss)+" | Val Acc = "+str(current_val_acc))

        if current_val_acc >= val_acc:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_val.torch"))
            tqdm.write("Current val best found at epoch="+str(epoch+1)+" with acc = "+str(current_val_acc))
            val_acc = current_val_acc 
            best_val_model = copy.deepcopy(model)

    tqdm.write("--------- Done Training ---------")

    tqdm.write("--------- Running Evaluation ---------")
    las, uas = util.get_accuracy_cm(model=best_val_model, data_loader=test_loader, parse_set=parse_set, \
        reverse_tagset=reverse_tagset, device=cfg.DEVICE)
    tqdm.write("Test Accuracies : las="+str(las)+", uas="+str(uas))

    tqdm.write("--------- Done ---------")


    
    
    
    

    