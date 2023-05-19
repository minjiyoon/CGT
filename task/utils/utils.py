import csv
import networkx as nx
import numpy as np
import random
import scipy.sparse as sp
from collections import defaultdict
from sklearn import metrics
from sklearn.preprocessing import normalize
from os.path import exists

import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

train_ratio = 0.4
val_ratio = 0.2
def split_ids(args, graph, labels):
    node_ids = list(range(graph.shape[0]))
    random.shuffle(node_ids)

    ids = {}
    ids['train'] = node_ids[:int(train_ratio * len(node_ids))]
    ids['val'] = node_ids[int(train_ratio * len(node_ids)):int((train_ratio + val_ratio) * len(node_ids))]
    ids['test'] = node_ids[int((train_ratio + val_ratio) * len(node_ids)):]

    return ids

def load_ogbn(args):
    master_file = args.data_dir + "/ogbn-master.csv"
    master = pd.read_csv(master_file, index_col = 0)
    meta_dict = master[args.dataset]

    add_inverse_edge = meta_dict['add_inverse_edge'] == 'True'
    binary = meta_dict['binary'] == 'True'
    additional_node_files = []
    additional_edge_files = []

    data_dir = args.data_dir + "/" + args.dataset + "/"
    data = read_graph_pyg(data_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=binary)[0]
    data.x = normalize_features(data.x)
    node_feat = data.x.numpy()

    data.edge_index = to_undirected(data.edge_index)
    graph = convert_to_edge_list(data.edge_index, node_feat)

    label = pd.read_csv(osp.join(data_dir, 'node-label.csv.gz'), compression='gzip', header = None).values
    label = label.squeeze()

    feat_size = node_feat.shape[1]
    label_size = label.max() - label.min() + 1

    return graph, node_feat, label, feat_size, label_size


def load_graph(args):
    if args.dataset in ("ogbn-arxiv", "ogbn-products"):
        return load_ogbn(args)

    dataset = args.data_dir + "/" + args.dataset + ".npz"
    with np.load(dataset, allow_pickle = True) as loader:
        loader = dict(loader)

        # Adjacency matrix
        graph = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])
        graph = graph + graph.transpose()
        if args.noise_num > 0:
            graph = graph + sp.identity(graph.shape[0])
        graph = sp.csr_matrix.toarray(graph)

        # Feature matrix
        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape'])
            features = sp.csr_matrix.toarray(features)
            # Normalize
            features = normalize(features, axis=1, norm='l2')
            #features = features - np.mean(features, axis=0)
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            features = loader['attr_matrix']
        else:
            features = None

        # Labels
        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']), shape=loader['labels_shape'])
            labels = sp.csr_matrix.toarray(labels)
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

    feat_size = features.shape[1]
    labels = labels - labels.min()
    label_size = labels.max() - labels.min() + 1

    return graph, features, labels, feat_size, label_size


def calc_loss(y_pred, y_true):
    if len(y_pred.shape) == 1:
        y_pred = torch.unsqueeze(y_pred, 0)
    if len(y_true.shape) == 2:
        y_true = torch.squeeze(y_true)
    loss_train = F.cross_entropy(y_pred, y_true)
    return loss_train


def calc_f1(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    y_true = y_true.cpu()
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")


