import numpy as np
import torch
import os
import json
import scripts.state as state


from torch.utils.data import Dataset
from torch.nn import functional as F

def get_file_contents(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def get_train_data(file_path, split="train"):
    parse_data = []
    with open(file_path) as f:
        data = f.readlines()
        for line in data:
            parse_dict = {}
            full_line = line.split("|||")
            parse_dict["words"] = full_line[0].split()
            parse_dict["pos"] = full_line[1].split()
            if split == "train" or split == "val" or split == "test":
                parse_dict["tags"] = full_line[2].split()
            parse_data.append(parse_dict)
    return parse_data

def convert_to_labels(tokens):
    label_dict = {}
    for i in range(len(tokens)):
        label_dict[tokens[i]] = i
    return label_dict

def convert_sentence_to_labels(tokens):
    # label_dict = {}
    # for i in range(1, len(tokens)+1):
    #     label_dict[tokens[i]] = i
    return convert_to_labels(tokens)

def custom_collate_fn(data_dict):
    return data_dict

def get_stack(stack):
    if len(stack) == 0:
        sw = ['[PAD]', '[PAD]'] 
        sq = ['NULL', 'NULL']
    elif len(stack) == 1:
        sw = [stack[-1].word, '[PAD]']
        sq = [stack[-1].pos, 'NULL']
    else:
        sw = [stack[-1].word, stack[-2].word] 
        sq = [stack[-1].pos, stack[-2].pos]
    return sw, sq

def get_buffer(buffer):
    if len(buffer) == 0:
        sw = ['[PAD]', '[PAD]'] 
        sq = ['NULL', 'NULL']
    elif len(buffer) == 1:
        sw = [buffer[0].word, '[PAD]']
        sq = [buffer[0].pos, 'NULL']
    else:
        sw = [buffer[0].word, buffer[1].word] 
        sq = [buffer[0].pos, buffer[1].pos]
    return sw, sq


class ParsingDataset(Dataset):
    def __init__(self, data_file, pose_set, tagset, glove, split="train"):
        super(ParsingDataset, self).__init__()
        self.pose_set = pose_set
        self.tagset = tagset
        self.data_list = get_train_data(data_file, split)
        self.processed_data = self.pre_process_data(self.data_list)
        self.glove = glove
        
    def __len__(self):
        return len(self.processed_data)    
    
    def pre_process_data(self, data_list):
        processed_data = []
        
        for parse_data in data_list:
            words_labels = convert_sentence_to_labels(parse_data["words"])
            init_stack = []
            init_buffer = [state.Token(idx=words_labels[parse_data["words"][i]], word=parse_data["words"][i], pos=parse_data["pos"][i]) for i in range(len(parse_data["words"]))]
            init_dependencies = []
            ps = state.ParseState(stack=init_stack, parse_buffer=init_buffer, dependencies=init_dependencies)

            for i in range(len(parse_data["tags"])):
                current_action = parse_data["tags"][i]

                sw, sp = get_stack(ps.stack)
                bw, bp = get_buffer(ps.parse_buffer)

                in_words = sw + bw
                in_pos = sp + bp
                
                processed_data_instance = {"words":in_words, "pos":in_pos, "action":current_action}
                processed_data.append(processed_data_instance)

                if "REDUCE_L" in current_action:
                    state.left_arc(ps, current_action.removeprefix('REDUCE_L'))
                elif "REDUCE_R" in current_action:
                    state.right_arc(ps, current_action.removeprefix('REDUCE_R'))
                else:
                    state.shift(ps)
                    
        return processed_data


    def __getitem__(self, idx):
        current_data = self.processed_data[idx].copy()
        words = self.glove.get_vecs_by_tokens(current_data["words"])
        pos_labels = torch.tensor(np.array([self.pose_set[p] for p in current_data["pos"]]))
        action_label = torch.tensor(np.array(self.tagset[current_data["action"]]))
        return words, pos_labels, action_label
    


class ParsingDatasetEval(Dataset):
    def __init__(self, data_file, pose_set, tagset, glove, split="train"):
        super(ParsingDatasetEval, self).__init__()
        self.pose_set = pose_set
        self.tagset = tagset
        self.data_list = get_train_data(data_file, split)
        self.glove = glove
        
    def __len__(self):
        return len(self.data_list)    

    def __getitem__(self, idx):
        current_data = self.data_list[idx].copy()
        return current_data
            
def get_child_data(child, ldict):
    words = ['[PAD]', '[PAD]']
    pos = ['NULL', 'NULL']
    labels = ['NULL', 'NULL']
    if len(child) > 1:
        min_idx = min(child,key=lambda x:x.idx)
        max_idx = min(child,key=lambda x:x.idx)
        words = [min_idx.word, max_idx.word]
        pos = [min_idx.pos, max_idx.pos]
        labels = [ldict[w] for w in words]    
    return  words, pos, labels
           
            
def get_children(ps, sw):
    ch0 = []
    ch1 = []
    ldict = {}
    for d in ps.dependencies:
        if d.source == sw[0]:
            ch0.append(d.target)
            ldict[d.target.word] = d.label
        if d.source == sw[1]:
            ch1.append(d.target)
            ldict[d.target.word] = d.label

    w0, p0, l0 = get_child_data(ch0, ldict)
    w1, p1, l1 = get_child_data(ch1, ldict)
    wc = w0 + w1
    pc = p0 + p1
    lc = l0 + l1
    
    return wc, pc, lc

def get_parse_labels(file_path):
    actions = get_file_contents(file_path)
    pl = []
    for current_action in actions:
        if "REDUCE" in current_action:
            pl.append(current_action[9:])
    pl = list(set(pl))
    pl.append('NULL')
    return pl

class ParsingDataset2(Dataset):
    def __init__(self, data_file, pose_set, tagset, parse_set, glove, split="train"):
        super(ParsingDataset2, self).__init__()
        self.pose_set = pose_set
        self.tagset = tagset
        self.parse_set = parse_set
        self.data_list = get_train_data(data_file, split)
        self.processed_data = self.pre_process_data(self.data_list)
        self.glove = glove
        
    def __len__(self):
        return len(self.processed_data)    
    
    def pre_process_data(self, data_list):
        processed_data = []
        
        for parse_data in data_list:
            words_labels = convert_sentence_to_labels(parse_data["words"])
            init_stack = []
            init_buffer = [state.Token(idx=words_labels[parse_data["words"][i]], word=parse_data["words"][i], pos=parse_data["pos"][i]) for i in range(len(parse_data["words"]))]
            init_dependencies = []
            ps = state.ParseState(stack=init_stack, parse_buffer=init_buffer, dependencies=init_dependencies)

            for i in range(len(parse_data["tags"])):
                current_action = parse_data["tags"][i]

                sw, sp = get_stack(ps.stack)
                bw, bp = get_buffer(ps.parse_buffer)

                wc, pc, lc = get_children(ps, sw)
                
                in_words = sw + bw + wc
                in_pos = sp + bp + pc
                
                
                processed_data_instance = {"words":in_words, "pos":in_pos, "action":current_action, "ps":lc}
                processed_data.append(processed_data_instance)

                if "REDUCE_L" in current_action:
                    state.left_arc(ps, current_action[9:])
                elif "REDUCE_R" in current_action:
                    state.right_arc(ps, current_action[9:])
                else:
                    state.shift(ps)
                    
        return processed_data


    def __getitem__(self, idx):
        current_data = self.processed_data[idx].copy()
        words = self.glove.get_vecs_by_tokens(current_data["words"])
        pos_labels = torch.tensor(np.array([self.pose_set[p] for p in current_data["pos"]]))
        parse_labels = torch.tensor(np.array([self.parse_set[p] for p in current_data["ps"]]))
        action_label = torch.tensor(np.array(self.tagset[current_data["action"]]))
        return words, pos_labels, parse_labels, action_label


if __name__ == "__main__":
    import config as cfg
    import torchtext
    from scripts.evaluate import compute_metrics, get_deps
    
    pose_set = convert_to_labels(get_file_contents(cfg.pose_set_file))
    tagset = convert_to_labels(get_file_contents(cfg.tagset_file))
    glove = torchtext.vocab.GloVe(name=cfg.glove_name, dim=cfg.glove_dim)
    a = ParsingDataset(data_file=cfg.train_data_path, pose_set=pose_set, tagset=tagset, glove=glove, split='train')
    
    words_lists = []
    actions_lists = []
    
    data_list = a.data_list
    for parse_data in data_list:
        words_lists.append(parse_data["words"])
        actions_lists.append(parse_data["tags"])
    
    uas, las = compute_metrics(words_lists, actions_lists, actions_lists)  
    print(uas, las)
    
    # all_deps = get_deps(words_lists, actions_lists, 2)

    