import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datetime import datetime
import numpy as np
import torch
from operator import itemgetter
from time import perf_counter

from args import get_args
from task.utils.utils import load_graph
from task.utils.sample import split_ids, split_dataset, graph_sampler

from task.aggregation.gcn import run as aggregation
import generator.gpt.gpt as gpt


def evaluate(args, feats, graphs, labels, ids, supp, feature_matrix=None):
    acc_mic = np.zeros(len(args.model_list))
    acc_mac = np.zeros(len(args.model_list))
    train_time = np.zeros(len(args.model_list))
    test_time = np.zeros(len(args.model_list))

    feats, graphs, labels, ids = split_dataset(feats, graphs, labels, ids)

    for i, model_name in enumerate(args.model_list):
        if args.task_name == "aggregation":
            acc_mic[i], acc_mac[i], train_time[i], test_time[i] = aggregation(args, model_name, feats, graphs, labels, ids, feature_matrix)
    return acc_mic, acc_mac


def main():
    args = get_args()

    adj, feat, label, feat_size, label_size = load_graph(args)
    ids, supp = split_ids(args, adj, label)
    args.feat_size = feat_size
    args.label_size = label_size
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trials = 3
    train_acc_list = np.zeros((3, len(args.model_list), trials))
    test_acc_list = np.zeros((3, len(args.model_list), trials))

    for t in range(trials):

        # Check GNN performance on the original dataset
        # Sample computation graphs for each nodes: list of feature matrices, adjacency matrices, labels
        start_time = perf_counter()
        feat_list, graph_list, label_list, lid_list = graph_sampler(args, adj, feat, label, ids)
        print('original sampling time: {:.3f}'.format(perf_counter() - start_time))

        start_time = perf_counter()
        if args.org_code:
            temp_feat = torch.tensor(feat)
        else:
            temp_feat = torch.cat((torch.tensor(feat), torch.zeros(1, feat.shape[1])), dim=0)
        train_acc, test_acc = evaluate(args, feat_list, graph_list, label_list, lid_list, supp, feature_matrix=temp_feat)
        train_acc_list[0, : , t] = train_acc
        test_acc_list[0, : , t] = test_acc
        print('original evaluation time: {:.3f}, acc: {}'.format(perf_counter() - start_time, test_acc))

        # Train GPT on the original feature list
        # GPT will output the clustered feature list and synthetic feature lists
        start_time = perf_counter()
        generated_list, cluster_center = gpt.run(args, adj, feat, label, ids, train_name=args.gpt_train_name)
        print('GPT total time: {:.3f}'.format(perf_counter() - start_time))

        # Check GNN performance on the clustered dataset
        start_time = perf_counter()
        train_acc, test_acc = evaluate(args, **generated_list['cluster'], supp=supp, feature_matrix=cluster_center)
        train_acc_list[1, : , t] = train_acc
        test_acc_list[1, : , t] = test_acc
        print('cluster evaluation time: {:.3f}, acc: {}'.format(perf_counter() - start_time, test_acc))

        # Check GNN performance on the generated dataset
        start_time = perf_counter()
        train_acc, test_acc = evaluate(args, **generated_list['generate'], supp=supp, feature_matrix=cluster_center)
        train_acc_list[2, : , t] = train_acc
        test_acc_list[2, : , t] = test_acc
        print('synthetic evaluation time: {:.3f}, acc: {}'.format(perf_counter() - start_time, test_acc))

        del feat_list
        del graph_list
        del label_list
        del lid_list
        del generated_list
        gc.collect()

    test_acc_avg = np.average(test_acc_list, axis=2)
    test_acc_std = np.std(test_acc_list, axis=2)
    version_list = ['ori', 'cls', 'gen']

    print('\nTask: ' + args.task_name)
    print('\nModel: ' + args.gpt_model + ', Dataset: ' + args.dataset)
    for model_name in args.model_list:
        print(model_name, end=', ')
    for model_id in range(len(args.model_list)):
        print("\nORI: {:.2f} {:.3f}, CLS: {:.2f} {:.3f}, GEN: {:.2f} {:.3f}".format(test_acc_avg[0][model_id], test_acc_std[0][model_id],\
                                                            test_acc_avg[1][model_id], test_acc_std[1][model_id],\
                                                            test_acc_avg[2][model_id], test_acc_std[2][model_id],), end='')


if __name__ == "__main__":
    main()

