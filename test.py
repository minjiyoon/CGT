import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
from operator import itemgetter
from datetime import datetime
from time import perf_counter

from args import get_args
from task.utils.dataset import Dataset, collate
from task.utils.utils import load_graph, split_ids

from task.aggregation.gcn import run as aggregation
import generator.gpt.gpt as gpt


def evaluate(args, train_set, val_set, test_set):
    acc_mic = np.zeros(len(args.model_list))
    acc_mac = np.zeros(len(args.model_list))

    params = {'batch_size': args.batch_size,
              'num_workers': args.num_workers,
              'prefetch_factor': args.prefetch_factor,
              'collate_fn': collate,
              'shuffle': True,
              'drop_last': True}
    train_loader = torch.utils.data.DataLoader(train_set, **params)
    val_loader = torch.utils.data.DataLoader(val_set, **params)
    test_loader = torch.utils.data.DataLoader(test_set, **params)

    for i, model_name in enumerate(args.model_list):
        if args.task_name == "aggregation":
            acc_mic[i], acc_mac[i] = aggregation(args, model_name, train_loader, val_loader, test_loader)
    return acc_mic, acc_mac


def main():
    args = get_args()
    args.gpt_train_name = args.task_name + '_' + args.dataset + datetime.now().strftime("_%Y%m%d_%H%M%S")

    # Load the original graph datasets
    adj, feat, label, feat_size, label_size = load_graph(args)
    ids = split_ids(args, adj, label)
    args.feat_size = feat_size
    args.label_size = label_size
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trials = 1
    acc_mic_list = np.zeros((2, len(args.model_list), trials))
    acc_mac_list = np.zeros((2, len(args.model_list), trials))

    for t in range(trials):

        # Prepare duplicate-encoded computation graphs
        train_set = Dataset(args, "train", adj, feat, label, ids)
        val_set = Dataset(args, "val", adj, feat, label, ids)
        test_set = Dataset(args, "test", adj, feat, label, ids)
        # Check GNN performance on the original dataset
        start_time = perf_counter()
        acc_mic, acc_mac = evaluate(args, train_set, val_set, test_set)
        acc_mic_list[0, : , t] = acc_mic
        acc_mac_list[0, : , t] = acc_mac
        print('Original evaluation time: {:.3f}, acc: {}'.format(perf_counter() - start_time, acc_mic))

        ## Train GPT on the original graph
        start_time = perf_counter()
        gen_train_set, gen_val_set, gen_test_set = gpt.run(args, adj, feat, label, ids)
        print('GPT training/generation total time: {:.3f}'.format(perf_counter() - start_time))

        ## Check GNN performance on the generated dataset
        start_time = perf_counter()
        acc_mic, acc_mac = evaluate(args, gen_train_set, gen_val_set, gen_test_set)
        acc_mic_list[0, : , t] = acc_mic
        acc_mac_list[0, : , t] = acc_mac
        print('Synthetic evaluation time: {:.3f}, acc: {}'.format(perf_counter() - start_time, acc_mic))

    test_acc_avg = np.average(acc_mic_list, axis=2)
    test_acc_std = np.std(acc_mic_list, axis=2)

    print('\nTask: ' + args.task_name + ', Dataset: ' + args.dataset)
    for model_name in args.model_list:
        print(model_name, end=', ')
    for model_id in range(len(args.model_list)):
        print("\nORI: {:.2f} {:.3f}, GEN: {:.2f} {:.3f}".format(test_acc_avg[0][model_id], test_acc_std[0][model_id],\
                                                            test_acc_avg[1][model_id], test_acc_std[1][model_id]))


if __name__ == "__main__":
    main()

