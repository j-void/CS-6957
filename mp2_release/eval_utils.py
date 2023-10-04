import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict
import math
import scripts.state as state
import copy
from scripts.evaluate import compute_metrics
import data_loader as dl


def val(model, pose_set, glove, parse_data, device, reverse_tagset):
    words_labels = dl.convert_sentence_to_labels(parse_data["words"])
    init_stack = []
    init_buffer = [state.Token(idx=words_labels[parse_data["words"][i]], word=parse_data["words"][i], pos=parse_data["pos"][i]) for i in range(len(parse_data["words"]))]
    init_dependencies = []
    ps = state.ParseState(stack=init_stack, parse_buffer=init_buffer, dependencies=init_dependencies)
    

    pred_actions = []

    max_num_actions = 2*len(parse_data["words"]) - 1

    while state.is_final_state(ps, cwindow=2) == False:# and max_num_actions > 0:

        sw, sp = dl.get_stack(ps.stack)
        bw, bp = dl.get_buffer(ps.parse_buffer)

        in_words = sw + bw
        in_pos = sp + bp
        pos_labels = torch.tensor(np.array([pose_set[pos] for pos in in_pos])).unsqueeze(0).to(device)
        word_embeds = glove.get_vecs_by_tokens(in_words).unsqueeze(0).to(device)

        pred_action_probs = model(word_embeds, pos_labels)

        ## get top k predictions
        _ , top_indices = torch.topk(pred_action_probs, 2)
        top_indices = top_indices.squeeze()
        current_action = reverse_tagset[top_indices[0].item()]

        ## solve for illegal actions
        if "REDUCE" in current_action and len(ps.stack) <= 1:
            current_action = "SHIFT"
        elif "SHIFT" in current_action and len(ps.parse_buffer) < 1:
            current_action = reverse_tagset[top_indices[1].item()]

        if "REDUCE_L" in current_action:
            state.left_arc(ps, current_action[9:])
        elif "REDUCE_R" in current_action:
            state.right_arc(ps, current_action[9:])
        else:
            state.shift(ps)
            
        pred_actions.append(current_action)

        max_num_actions -= 1

    return pred_actions



def get_accuracy(model, data_loader, reverse_tagset, device):
    model.eval()

    with torch.no_grad():
        words_lists = []
        actions_lists = []
        pred_actions_list = []

        for parse_data in data_loader:
            pd = parse_data[0]
            pred_actions = val(model, data_loader.dataset.pose_set, data_loader.dataset.glove, pd, device, reverse_tagset)
            pred_actions_list.append(pred_actions)
            words_lists.append(pd["words"])
            actions_lists.append(pd["tags"])
        
        uas, las = compute_metrics(words_lists, actions_lists, pred_actions_list)                
    return las, uas

def run_hidden_data(model, data_loader, reverse_tagset, path, device, complete=True):
    model.eval()
    file = open(path, "w")
    with torch.no_grad():

        for parse_data in data_loader:
            pd = parse_data[0]
            pred_actions = val(model, data_loader.dataset.pose_set, data_loader.dataset.glove, pd, device, reverse_tagset)

            words_str = ' '.join(pd["words"])
            pos_str = ' '.join(pd["pos"])
            action_str = ' '.join(pred_actions)
            complete_str = ' ||| '.join([words_str, pos_str, action_str])
            if complete==True:
                file.write(complete_str+"\n")
            else:
                file.write(action_str+"\n")
            
    file.close()

