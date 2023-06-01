import numpy as np
import os
import random
import torch

from time import perf_counter
from sklearn.decomposition import PCA
from k_means_constrained import KMeansConstrained


def kmeans(feats, cluster_num, cluster_size, cluster_sample_num):
    if cluster_sample_num < feats.shape[0]:
        px_ids = random.sample(list(range(feats.shape[0])), cluster_sample_num)
        x = feats[px_ids]
    else:
        x = feats

    pca = PCA(n_components=min(feats.shape[1], 128))
    x_pca = pca.fit_transform(x)

    clf = KMeansConstrained(n_clusters=cluster_num, size_min=cluster_size, init='random', n_init=1, max_iter=8)
    clf.fit(x_pca)
    centers = pca.inverse_transform(clf.cluster_centers_)

    return centers


def cluster_feats(args, feats):
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
    cluster_centers = kmeans(feats, args.cluster_num, args.cluster_size, args.cluster_sample_num)

    # Cluster the original dataset
    batch_size = 1000
    cluster_ids = np.zeros(feats.shape[0])
    for batch in range(feats.shape[0] // batch_size + 1):
        if batch < feats.shape[0] // batch_size:
            idx = list(range(batch * batch_size, (batch + 1) * batch_size))
        else:
            idx = list(range(batch * batch_size, feats.shape[0]))
        cluster_ids[idx] = ((feats[idx, None, :] - cluster_centers[None, :, :]) ** 2).sum(-1).argmin(1)

    # Append empty_id
    cluster_ids = torch.LongTensor(np.append(cluster_ids, args.cluster_num))
    cluster_centers = torch.FloatTensor(np.concatenate((cluster_centers, np.zeros((1, cluster_centers.shape[1]))), axis=0))

    print("Clustering time: {:.3f}".format(perf_counter() - start_time))

    return cluster_ids, cluster_centers

