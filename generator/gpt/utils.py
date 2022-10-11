import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def compute_short_seq(sample_num, step_num):
    seq_id_list = []
    def recursion(layer, locat_list):
        for i in range(sample_num):
            new_locat_list = locat_list + [i]
            if layer == step_num:
                seq_id = [0, 1]
                abs = 2
                for j in range(1, step_num + 1):
                    new_id = abs + sample_num * new_locat_list[j-1] + new_locat_list[j]
                    seq_id.append(new_id)
                    abs += sample_num ** j
                seq_id_list.extend(seq_id)
            else:
                recursion(layer+1, new_locat_list)
    recursion(1, [0])

    comp_length = 2 # start node and root node
    for l in range(1, step_num + 1):
        comp_length += sample_num ** l
    rev_id_list = []
    for i in range(comp_length):
        rev_id_list.append(seq_id_list.index(i))

    return rev_id_list


@torch.no_grad()
def sample(model, x, lbl, temperature=1.0, sample=True, top_k=None):
    model.eval()

    gen_list = []
    comp_list = x
    batch_size = x.size(0)

    for step in range(model.step_num + 1):
        logits, _ = model(x, lbl)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        gen_list.append(ix)
        x = torch.concat((x, ix), dim=-1)
        x = x.repeat_interleave(model.sample_num, dim=0)
        lbl = lbl.repeat_interleave(model.sample_num)

    for step in range(model.step_num + 1):
        comp_list = torch.cat((comp_list, gen_list[step].view(batch_size, model.sample_num ** step)), dim=1)

    return comp_list

