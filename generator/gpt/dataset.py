import numpy as np
import random
from itertools import permutations

import torch

class DatasetType:
    """
    sequences include only parent nodes
    """

    def __init__(self, args, adjs, cluster_ids, labels, ids):
        self.step_num = args.subgraph_step_num
        self.sample_num = args.subgraph_sample_num
        self.noise_num = args.noise_num
        self.short_seq_num = (args.subgraph_sample_num + args.noise_num)**args.subgraph_step_num

        self.empty_id = args.cluster_num
        self.start_id = args.cluster_num + 1
        # cluster centers + start_id + empty_id
        self.vocab_size = args.cluster_num + 2
        # start_node + root_node + num_layers
        self.block_size = 1 + 1 + self.step_num

        self.adjs = adjs
        self.cluster_ids = cluster_ids
        self.labels = labels
        self.ids = ids

        self.compute_short_seq()
        self.reset()

    def compute_short_seq(self):
        self.seq_id_list = []
        sample_num = self.sample_num + self.noise_num
        def recursion(layer, locat_list):
            for i in range(sample_num):
                new_locat_list = locat_list + [i]
                if layer == self.step_num:
                    seq_id = [0, 1]
                    abs = 2
                    for j in range(1, self.step_num + 1):
                        new_id = abs + sample_num * new_locat_list[j-1] + new_locat_list[j]
                        seq_id.append(new_id)
                        abs += sample_num ** j
                    self.seq_id_list.extend(seq_id)
                else:
                    recursion(layer+1, new_locat_list)
        recursion(1, [0])

    def reset(self):
        self.pt_dataset = []
        self.label_dataset = []

        empty_org_id = self.cluster_ids.shape[0] - 1
        for seed_id in self.ids:
            label_id = self.labels[seed_id]
            sampled_cluster_ids = [self.start_id, self.cluster_ids[seed_id]]
            curr_target_list = [seed_id]

            for _ in range(self.step_num):
                new_target_list = []
                for target_id in curr_target_list:
                    # Get neighbor list
                    if target_id == empty_org_id:
                        source_ids = []
                    else:
                        source_ids = np.nonzero(self.adjs[target_id])[0].tolist()

                    # Sample fixed number of neighbors
                    if len(source_ids) == 0:
                        sampled_ids = self.sample_num * [empty_org_id]
                    elif len(source_ids) < self.sample_num:
                        sampled_ids = source_ids + (self.sample_num - len(source_ids)) * [empty_org_id]
                    else:
                        sampled_ids = np.random.choice(source_ids, self.sample_num, replace = False).tolist()

                    if self.noise_num > 0:
                        perm = np.random.permutation(self.adjs.shape[0])[:self.noise_num]
                        sampled_ids = np.concatenate((sampled_ids, perm), axis=0)

                    sampled_cluster_ids.extend(self.cluster_ids[sampled_ids])
                    new_target_list.extend(sampled_ids)

                curr_target_list = new_target_list

            self.pt_dataset.append(torch.LongTensor(sampled_cluster_ids))
            self.label_dataset.append(torch.tensor(label_id, dtype=torch.int64))

    def comp_ids(self):
        return torch.stack(self.pt_dataset, 0)[:, 1:]

    def __len__(self):
        return len(self.pt_dataset)

    def get(self, start_id, end_id):
        x = torch.stack(self.pt_dataset[start_id:end_id], dim=0)
        lbl = torch.stack(self.label_dataset[start_id:end_id], dim=0)
        x = x[:, self.seq_id_list].view(-1, self.block_size)
        lbl = lbl.repeat_interleave(self.short_seq_num)
        return x[:, :-1], x[:, 1:], lbl

