import torch
import numpy as np
from collections import defaultdict

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, split, adjs, feats, labels, ids):
        self.adjs = adjs
        self.feats = feats
        self.labels = labels
        self.ids = ids[split]

        self.empty_id = feats.shape[0]
        self.feats = np.concatenate((self.feats, np.zeros((1, feats.shape[1]))), axis=0)

        self.step_num = args.subgraph_step_num
        self.sample_num = args.subgraph_sample_num
        self.noise_num = args.noise_num
        self.self_connection = args.self_connection
        self.dup_adj = self.compute_dup_adj()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        seed_id = self.ids[index]
        sampled_nodes = [seed_id]
        curr_target_list = [seed_id]
        for _ in range(self.step_num):
            new_target_list = []
            for target_id in curr_target_list:
                # Get neighbor list
                if target_id == self.empty_id:
                    source_ids = []
                else:
                    source_ids = np.nonzero(self.adjs[target_id])[0].tolist()
                # Sample fixed number of neighbors
                if len(source_ids) == 0:
                    sampled_ids = self.sample_num * [self.empty_id]
                elif len(source_ids) < self.sample_num:
                    sampled_ids = source_ids + (self.sample_num - len(source_ids)) * [self.empty_id]
                else:
                    sampled_ids = np.random.choice(source_ids, self.sample_num, replace = False).tolist()

                if self.noise_num > 0:
                    perm = np.random.permutation(self.adjs.shape[0])[:self.noise_num]
                    sampled_ids = np.concatenate((sampled_ids, perm), axis=0)

                sampled_nodes.extend(sampled_ids)
                new_target_list.extend(sampled_ids)

            curr_target_list = new_target_list

        return {"feat": torch.FloatTensor(self.feats[sampled_nodes]),
                "adj": self.dup_adj,
                "label": torch.LongTensor([self.labels[seed_id]])
                }

    def compute_dup_adj(self):
        """ duplicate-encoded adjacency matrix (fixed shape)"""
        seed_id = 0
        sampled_nodes = [seed_id]
        curr_target_list = [seed_id]
        sampled_edges = defaultdict(list)
        for _ in range(self.step_num):
            new_target_list = []
            for target_id in curr_target_list:
                # Get neighbor list
                source_ids = list(range(len(sampled_nodes), \
                                    len(sampled_nodes) + self.sample_num + self.noise_num))

                sampled_nodes.extend(source_ids)
                new_target_list.extend(source_ids)
                sampled_edges[target_id].extend(source_ids)

            curr_target_list = new_target_list

        # Generate adjacency matrix
        rows = []
        cols = []
        for target_id in sampled_edges.keys():
            for source_id in sampled_edges[target_id]:
                rows.append(target_id)
                cols.append(source_id)

        # Define adjacency matrix
        indices = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)], dim=0)
        attention = torch.ones(len(cols))
        dense_shape = torch.Size([len(sampled_nodes), len(sampled_nodes)])
        sampled_adj = torch.sparse.FloatTensor(indices, attention, dense_shape).to_dense()

        # Remove zero_in_degree
        if self.self_connection:
            sampled_adj = sampled_adj + torch.diag(torch.ones(len(sampled_nodes)))

        return sampled_adj

def collate(items):
    items = [(item["feat"], item["adj"], item["label"]) for item in items]
    (feats, adjs, labels) = zip(*items)

    result= dict(
        feat = torch.cat(feats),
        adj = torch.stack(adjs, dim=0),
        label = torch.cat(labels)
        )
    return result
