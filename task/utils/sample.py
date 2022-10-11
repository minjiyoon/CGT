import csv
from collections import defaultdict
import os
import random
import re
import networkx as nx
import numpy as np
import torch

from ..shift.srgnn import biased_sample

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

train_ratio = 0.6
def split_ids(args, graph, labels):
    node_ids = list(range(graph.shape[0]))
    random.shuffle(node_ids)

    ids = {}
    ids['train'] = node_ids[:int(train_ratio * len(node_ids))]
    ids['test'] = node_ids[int(train_ratio * len(node_ids)):]

    supp = None
    if args.task_name == "shift" and args.alpha > 0:
        train_bias, train_iid, supp = biased_sample(args, graph, torch.from_numpy(labels), ids)
        # QQQ: I think we need to shuffle ids; they are in order of label ids
        random.shuffle(train_bias)
        random.shuffle(train_iid)
        ids['train'] = train_bias
        #ids['train_iid'] = train_iid

    return ids, supp

val_ratio = 0.2
def split_dataset(feats, graphs, labels, lids):
    train_num = int((1 - val_ratio) * len(feats['train']))
    train_feats, train_graphs, train_labels, train_lids = feats['train'][:train_num], graphs['train'][:train_num], labels['train'][:train_num], lids['train'][:train_num]
    val_feats, val_graphs, val_labels, val_lids = feats['train'][train_num:], graphs['train'][train_num:], labels['train'][train_num:], lids['train'][train_num:]

    feats['train'] = train_feats
    feats['val'] = val_feats
    graphs['train'] = train_graphs
    graphs['val'] = val_graphs
    labels['train'] = train_labels
    labels['val'] = val_labels
    lids['train'] = train_lids
    lids['val'] = val_lids

    return feats, graphs, labels, lids

def graph_sampler(args, adjs, feats, labels, ids):
    """
    Sample computation graphs for every nodes

    Return:
        feat_list: dict of list of feature matrices of each computation graph
        graph_list: dict of list of adjacency matrix of each computation graph
        label_list: dict of list of label of each computation graph
        ids: dict of node ids of training/validation/test set
    """

    feat_list = defaultdict(list)
    graph_list = defaultdict(list)
    label_list = defaultdict(list)
    lid_list = defaultdict(list)

    for split in ids.keys():
        seed_ids = ids[split]

        if args.org_code:
            feat, graph, label, label_id = graph_sampler_org(args, adjs, feats, labels, seed_ids)
        else:
            feat, graph, label, label_id = graph_sampler_dup(args, adjs, feats, labels, seed_ids)

        feat_list[split] = feat
        graph_list[split] = graph
        label_list[split] = label
        lid_list[split] = label_id

    return feat_list, graph_list, label_list, lid_list


# Computation graph sampler using original coding
def graph_sampler_org(args, adjs, feats, labels, ids):
    step_num = args.subgraph_step_num
    sample_num = args.subgraph_sample_num
    batch_num = len(ids) // args.batch_size

    feat_list = []
    graph_list = []
    label_list = []
    lid_list = []

    for b in range(batch_num):
        seed_ids = ids[b * args.batch_size : (b+1) * args.batch_size]

        sampled_nodes = set()
        sampled_edges = defaultdict(list)

        sampled_nodes.update(seed_ids)
        curr_target_list = seed_ids
        for _ in range(step_num):
            new_target_list = []
            for target_id in curr_target_list:
                source_ids = np.nonzero(adjs[target_id])[0]
                if source_ids.shape[0] == 0:
                    sampled_ids = np.array([])
                elif source_ids.shape[0] < sample_num:
                    sampled_ids = source_ids
                else:
                    sampled_ids = np.random.choice(source_ids, sample_num, replace = False)
                if args.noise_num > 0:
                    perm = np.random.permutation(adjs.shape[0])[:args.noise_num]
                    sampled_ids = np.concatenate((sampled_ids, perm), axis=0)
                sampled_nodes.update(sampled_ids)
                sampled_edges[target_id].extend(sampled_ids)
                new_target_list.extend(sampled_ids)
            curr_target_list = new_target_list

        # Indexing nodes
        index_ = {}
        for sampled_id in sampled_nodes:
            index_[sampled_id] = len(list(index_.keys()))

        sampled_lid = []
        for seed_id in seed_ids:
            sampled_lid.append(index_[seed_id])

        # Convert to torch object
        #sampled_feats = torch.FloatTensor(feats[list(sampled_nodes)])
        sampled_feats = torch.LongTensor(list(sampled_nodes))
        sampled_label = torch.LongTensor(labels[seed_ids])

        # Generate adjacency matrix
        rows = []
        cols = []
        for target_id in sampled_edges.keys():
            target_index_ = index_[target_id]
            for source_id in sampled_edges[target_id]:
                source_index_ = index_[source_id]
                rows.append(target_index_)
                cols.append(source_index_)

        # Define adjacency matrix
        indices = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)], dim=0)
        attention = torch.ones(len(cols))
        dense_shape = torch.Size([len(index_), len(index_)])
        sampled_adj = torch.sparse.FloatTensor(indices, attention, dense_shape).to_dense()

        # Remove zero_in_degree
        if args.self_connection:
            sampled_adj = sampled_adj + torch.diag(torch.ones(len(index_)))

        feat_list.append(sampled_feats)
        graph_list.append(sampled_adj)
        label_list.append(sampled_label)
        lid_list.append(sampled_lid)

    return feat_list, graph_list, label_list, lid_list


# Computation graph sampler using duplication coding
# batch_size = 1
def graph_sampler_dup(args, adjs, feats, labels, ids):
    step_num = args.subgraph_step_num
    sample_num = args.subgraph_sample_num

    # Empty feature vector
    empty_id = feats.shape[0]

    comp_ids = []
    for seed_id in ids:
        sampled_nodes = [seed_id]
        curr_target_list = [seed_id]

        for _ in range(step_num):
            new_target_list = []
            for target_id in curr_target_list:
                # Get neighbor list
                if target_id == empty_id:
                    source_ids = []
                else:
                    source_ids = np.nonzero(adjs[target_id])[0].tolist()
                # Sample fixed number of neighbors
                if len(source_ids) == 0:
                    sampled_ids = sample_num * [empty_id]
                elif len(source_ids) < sample_num:
                    sampled_ids = source_ids + (sample_num - len(source_ids)) * [empty_id]
                else:
                    sampled_ids = np.random.choice(source_ids, sample_num, replace = False).tolist()

                if args.noise_num > 0:
                    perm = np.random.permutation(adjs.shape[0])[:args.noise_num]
                    sampled_ids = np.concatenate((sampled_ids, perm), axis=0)

                sampled_nodes.extend(sampled_ids)
                new_target_list.extend(sampled_ids)

            curr_target_list = new_target_list
        comp_ids.append(torch.LongTensor(sampled_nodes))

    feats = torch.FloatTensor(feats)
    comp_ids = torch.stack(comp_ids, 0)
    feat_list, graph_list, label_list, lid_list = make_batch(args, comp_ids, labels[ids], feats)

    return feat_list, graph_list, label_list, lid_list


def make_batch(args, ids_list, label_list, C):
    batch_size = args.batch_size
    step_num = args.subgraph_step_num
    sample_num = args.subgraph_sample_num + args.noise_num

    batch_num = len(ids_list) // batch_size

    feat_batch_list = []
    graph_batch_list = []
    label_batch_list = []
    lid_batch_list = []
    for b in range(batch_num):
        ids_batch = ids_list[b*batch_size:(b+1)*batch_size]
        labels_batch = torch.LongTensor(label_list[b*batch_size:(b+1)*batch_size])

        # Indexing nodes and label indices
        empty_id = args.cluster_num
        index = []
        distinct_ids = []
        lid_batch = []
        for i in range(ids_batch.shape[0]):
            for j in range(ids_batch[i].shape[0]):
                id = ids_batch[i][j].item()
                if id != empty_id:
                    if j == 0:
                        lid_batch.append(len(distinct_ids))
                    index.append(len(distinct_ids))
                    distinct_ids.append(id)
                else:
                    if j == 0:
                        print("ERROR: root node is generated as empty...")
                        lid_batch.append(len(distinct_ids))
                    index.append(0)

        # Distinct features
        feats_batch = distinct_ids
        #feats_batch = C[distinct_ids]

        rows = []
        cols = []
        base_idx = 0
        for i in range(ids_batch.shape[0]):
            ids = ids_batch[i].tolist()
            for row in range(0, len(ids) - sample_num**step_num):
                if ids_batch[i][row] == empty_id:
                    continue

                for col in range(1 + sample_num * row, 1 + sample_num * (row + 1)):
                    if ids_batch[i][col] == empty_id:
                        continue
                    rows.append(index[base_idx + row])
                    cols.append(index[base_idx + col])

            base_idx += len(ids)

        # Define adjacency matrix
        indices = torch.stack([torch.tensor(rows), torch.tensor(cols)], dim=0).type(torch.LongTensor)
        attention = torch.ones(len(cols))
        dense_shape = torch.Size([len(distinct_ids), len(distinct_ids)])
        adj_batch = torch.sparse.FloatTensor(indices, attention, dense_shape).to_dense()

        # Remove zero_in_degree
        if args.self_connection:
            adj_batch = adj_batch + torch.diag(torch.ones(len(distinct_ids)))

        feat_batch_list.append(feats_batch)
        graph_batch_list.append(adj_batch)
        label_batch_list.append(labels_batch)
        lid_batch_list.append(lid_batch)

    return feat_batch_list, graph_batch_list, label_batch_list, lid_batch_list

