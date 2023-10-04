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
torch.random.manual_seed(100)

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
    
    glove = torchtext.vocab.GloVe(name=cfg.glove_name, dim=cfg.glove_dim)

    test_dataset = ParsingDatasetEval(data_file=cfg.test_data_path, pose_set=pose_set, tagset=tagset, glove=glove, split='test')

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available()
	)
    
    hidden_dataset = ParsingDatasetEval(data_file=cfg.hidden_data_path, pose_set=pose_set, tagset=tagset, glove=glove, split='hidden')

    hidden_loader = DataLoader(
        hidden_dataset,
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

    model = TDParser2(pose_set_dim=len(pose_set), pos_embedding_dim=cfg.pos_embedding_dim, glove_dim=cfg.glove_dim, tagset_dim=len(tagset), combine=cfg.combine_type)
    model.to(cfg.DEVICE)
    model.load_state_dict(torch.load(os.path.join(save_dir, "model_val.torch")))
    
    torch.cuda.empty_cache()

    las, uas = util.get_accuracy(model=model, data_loader=test_loader, reverse_tagset=reverse_tagset, device=cfg.DEVICE)
    print(f"Test Accuracies : las={las}, uas={uas}")
    
    # tqdm.write("--------- Writing Hidden File ---------")
    # util.run_hidden_data(model=model, data_loader=hidden_loader, reverse_tagset=reverse_tagset, path=os.path.join(save_imdt, "hidden_output.txt"), device=cfg.DEVICE)
    
    # tqdm.write("--------- Done ---------")



    
    
    
    

    