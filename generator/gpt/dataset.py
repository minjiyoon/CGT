from collections import defaultdict
from itertools import permutations
import numpy as np
import random
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, adjs, cluster_ids, labels, ids):
        self.adjs = adjs
        self.adjs_list = isinstance(adjs, list)
        self.cluster_ids = cluster_ids
        self.labels = labels
        self.ids = ids

        self.step_num = args.subgraph_step_num
        self.sample_num = args.subgraph_sample_num
        self.noise_num = args.noise_num
        self.total_sample_num = self.sample_num + self.noise_num
        self.short_seq_num = (self.sample_num + self.noise_num) ** self.step_num

        self.empty_id = args.cluster_num
        self.start_id = args.cluster_num + 1
        self.vocab_size = args.cluster_num + 2 # cluster centers + start_id + empty_id
        self.block_size = 1 + 1 + self.step_num # start_node + root_node + num_layers

        self.compute_short_seq()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        org_empty_id = self.cluster_ids.shape[0] - 1
        seed_id = self.ids[index]
        sampled_cluster_ids = [self.start_id, self.cluster_ids[seed_id]]
        curr_target_list = [seed_id]
        for _ in range(self.step_num):
            new_target_list = []
            for target_id in curr_target_list:
                # Get neighbor list
                if target_id == org_empty_id:
                    source_ids = []
                else:
                    if self.adjs_list:
                        source_ids = self.adjs[target_id]
                    else:
                        source_ids = np.nonzero(self.adjs[target_id])[0].tolist()
                # Sample fixed number of neighbors
                if len(source_ids) == 0:
                    sampled_ids = self.sample_num * [org_empty_id]
                elif len(source_ids) < self.sample_num:
                    sampled_ids = source_ids + (self.sample_num - len(source_ids)) * [org_empty_id]
                else:
                    sampled_ids = np.random.choice(source_ids, self.sample_num, replace = False).tolist()

                if self.noise_num > 0:
                    perm = np.random.permutation(len(self.adjs))[:self.noise_num]
                    sampled_ids = np.concatenate((sampled_ids, perm), axis=0)

                sampled_cluster_ids.extend(self.cluster_ids[sampled_ids])
                new_target_list.extend(sampled_ids)

            curr_target_list = new_target_list

        return {"ids": torch.LongTensor(sampled_cluster_ids),
                "label": torch.LongTensor([self.labels[seed_id]])
                }

    def collate(self, items):
        items = [(item["ids"], item["label"]) for item in items]
        (idses, labels) = zip(*items)

        idses = torch.stack(idses, dim=0)
        labels = torch.stack(labels, dim=0)
        idses = idses[:, self.seq_id_list].view(-1, self.block_size)
        labels = labels.repeat_interleave(self.short_seq_num)

        result= dict(
            query = idses[:, :-1],
            predict = idses[:, 1:],
            label = labels
        )
        return result

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
                        new_id = abs + sample_num * new_locat_list[j - 1] + new_locat_list[j]
                        seq_id.append(new_id)
                        abs += sample_num ** j
                    self.seq_id_list.extend(seq_id)
                else:
                    recursion(layer + 1, new_locat_list)
        recursion(1, [0])


class QuantizedDataset(torch.utils.data.Dataset):
    def __init__(self, args, sequences, labels, cluster_centers):
        self.sequences = sequences
        self.labels = labels
        self.cluster_centers = cluster_centers

        self.step_num = args.subgraph_step_num
        self.sample_num = args.subgraph_sample_num + args.noise_num
        self.self_connection = args.self_connection
        self.dup_adj = self.compute_dup_adj()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return {"feat": self.cluster_centers[self.sequences[index]],
                "adj": self.dup_adj,
                "label": torch.LongTensor([self.labels[index]])
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
                source_ids = list(range(len(sampled_nodes), len(sampled_nodes) + self.sample_num))

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


