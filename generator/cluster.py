import os
import random
import numpy as np
from sklearn.decomposition import PCA
import torch
from k_means_constrained import KMeansConstrained
from time import perf_counter


def kmeans(feats, cluster_num, cluster_size, cluster_sample_num):
    if cluster_sample_num < feats.shape[0]:
        px_ids = random.sample(list(range(feats.shape[0])), cluster_sample_num)
        x = feats[px_ids]
    else:
        x = feats

    pca = PCA(n_components=128)
    x_pca = pca.fit_transform(x)

    clf = KMeansConstrained(n_clusters=cluster_num, size_min=cluster_size, init='random', n_init=1, max_iter=8)
    clf.fit(x_pca)
    centers = pca.inverse_transform(clf.cluster_centers_)

    return centers


def cluster_feat_list(args, feats, train_name):
    """
    Cluster feature vectors

    Input:
        org_feats: original feature matrices
    Return:
        cluster_ids: list of cluster ids where each feature belongs to
        cluster_centers: centers of clusters

    """
    # Define cluster centers
    start_time = perf_counter()
    C = kmeans(feats, args.cluster_num, args.cluster_size, args.cluster_sample_num)
    print("Clustering time: {:.3f}".format(perf_counter() - start_time))

    # Clustering original dataset
    start_time = perf_counter()
    batch_size = 1000
    cluster_ids = np.zeros(feats.shape[0])
    for batch in range(feats.shape[0] // batch_size + 1):
        if batch < feats.shape[0] // batch_size:
            idx = list(range(batch * batch_size, (batch+1) * batch_size))
        else:
            idx = list(range(batch * batch_size, feats.shape[0]))
        cluster_ids[idx] = ((feats[idx, None, :] - C[None, :, :])**2).sum(-1).argmin(1)

    # Append empty_id
    cluster_ids = torch.LongTensor(np.append(cluster_ids, args.cluster_num))
    C = torch.FloatTensor(np.concatenate((C, np.zeros((1, C.shape[1]))), axis=0))

    print("Clustered graph generation time: {:.3f}".format(perf_counter() - start_time))

    return cluster_ids, C


def map_back_to_features(ids, C):
    """
    Map cluster ids to feature vectors
    """
    feat_list = []

    empty_id = C.size(0)
    start_time = perf_counter()
    for i in range(len(ids)):
        feats = torch.zeros((ids[i].shape[0], C.shape[1]))
        nonzeros = (ids[i] != empty_id).nonzero().squeeze()
        feats[nonzeros] = C[ids[i][nonzeros]]
        feat_list.append(feats)

    return feat_list


