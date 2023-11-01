import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


@torch.no_grad()
def sample(model, x, lbl, temperature=1.0, sample=True, top_k=None):
    """
    Sample nodes in sequence which will be reconstructed into a computation graph.
    Args:
        model: CGT model
        x: seed nodes
        lbl: seed labels
        temperature: temperature of softmax
        sample: whether to sample or take the most likely
        top_k: k for top-k sampling
    """
    model.eval()

    generated_ids = []
    complete_ids = x
    batch_size = x.size(0)

    for step in range(model.step_num + 1):
        logits, _ = model(x, lbl)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        ix = torch.multinomial(probs, num_samples=1)

        generated_ids.append(ix)
        x = torch.cat((x, ix), dim=-1)
        x = x.repeat_interleave(model.sample_num, dim=0)
        lbl = lbl.repeat_interleave(model.sample_num)

    for step in range(model.step_num + 1):
        complete_ids = torch.cat((complete_ids, generated_ids[step].view(batch_size, model.sample_num ** step)), dim=1)

    return complete_ids

