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
import evaluate
import pandas as pd
import torchvision


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
 
 

def run_eval(model, dataloader, cer_metric, tokenizer):
    model.eval()
    mean_cer = 0.0
    with torch.no_grad():
        for pixel_values, labels, attention_mask in dataloader:
            pixel_values = pixel_values.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            pred_idx = model.generate(pixel_values)
            pred_str = tokenizer.batch_decode(pred_idx, skip_special_tokens=True)
            # labels[labels == -100] = tokenizer.pad_token_id
            label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # print("pred_str", pred_str)
            # print("label_str", label_str)
            cer = cer_metric.compute(predictions=pred_str, references=label_str)
            mean_cer += cer

    return mean_cer / len(dataloader)


def run_test(model, dataloader, cer_metric, tokenizer, save_dir):
    model.eval()
    mean_cer = 0.0
    out_dict = defaultdict(list)
    with torch.no_grad():
        for idx, (pixel_values, labels, attention_mask) in enumerate(dataloader):
            pixel_values = pixel_values.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            pred_idx = model.generate(pixel_values)
            pred_str = tokenizer.batch_decode(pred_idx, skip_special_tokens=True)
            # labels[labels == -100] = tokenizer.pad_token_id
            label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
            cer = cer_metric.compute(predictions=pred_str, references=label_str)
            mean_cer += cer
            name = "file_"+str(idx).zfill(5)
            PILImg = torchvision.transforms.functional.to_pil_image(pixel_values.squeeze())
            PILImg.save(os.path.join(save_dir, name+".png"))
            out_dict['name'].append(name)
            out_dict['gt'].append(label_str)
            out_dict['pred'].append(pred_str)
            
    df = pd.DataFrame(out_dict)
    df.to_csv(os.path.join(save_dir, "data.csv"), index=False)
    
    return mean_cer / len(dataloader)

if __name__ == "__main__":
    
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
    
    train_dataset = IAM_SROIE_Dataset(split="train", tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
	)

    val_dataset = train_dataset = IAM_SROIE_Dataset(split="val", tokenizer=tokenizer)

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
	)
    
    test_dataset = train_dataset = IAM_SROIE_Dataset(split="test", tokenizer=tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
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
    model = ViTEncoderLSTMDecoder(vocab_size=tokenizer.vocab_size)
    model.to(cfg.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), cfg.learning_rate)
    optimizer.zero_grad()

    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    

    min_val_cer = math.inf 
    cer_metric = evaluate.load("cer")
    
    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        loss_list_train = []        
        
        for idx, (pixel_values, labels, attention_mask) in enumerate((tqdm(train_loader, desc="Batch: "))):
            optimizer.zero_grad()
            pixel_values = pixel_values.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            attention_mask = attention_mask.to(cfg.DEVICE)
            pred_probs = model(pixel_values)
            # print(pred_probs.shape, attention_mask.view(-1).shape)
            final_loss = ce_loss(pred_probs, labels.view(-1))
            final_loss.backward()
            optimizer.step()

            loss_list_train.append(final_loss.item())
                    

        mean_train_loss = np.mean(np.array(loss_list_train))
        val_cer = run_eval(model=model, dataloader=val_loader, cer_metric=cer_metric, tokenizer=tokenizer)

        tqdm.write("Epoch - "+str(epoch+1)+" : Train Loss = "+str(mean_train_loss)+" | Val CER = "+str(val_cer))

        if min_val_cer >= val_cer:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_val.torch"))
            tqdm.write("Current val best found at epoch="+str(epoch+1))
            min_val_cer = val_cer 
            best_val_model = copy.deepcopy(model)

    tqdm.write("--------- Done Training ---------")
    
    test_cer = run_test(best_val_model, test_loader, cer_metric, tokenizer, save_imdt)
    tqdm.write("CER on Test Set = "+ str(test_cer))


    
    
    

    