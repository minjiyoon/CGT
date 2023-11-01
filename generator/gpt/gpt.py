import math
import os
import torch
from tqdm import tqdm
from time import perf_counter

from generator.gpt.dataset import Dataset, QuantizedDataset
from generator.gpt.model import GPTConfig, XLNet
from generator.gpt.trainer import Trainer, TrainerConfig
from generator.gpt.utils import sample
from generator.cluster import cluster_feats


def train(args, adjs, cluster_ids, labels, ids, split):
    """
    Train CGT
    Args:
        args: arguments
        adjs: a list of adjacency matrix of each computation graphs
        cluster_ids: cluster ids of each nodes
        labels: label of each nodes
        ids: node id list
        split: split name ('train', 'val', 'test')
    """
    # hyperparameters of computation graphs
    dataset = Dataset(args, adjs, cluster_ids, labels, ids)
    params = {'batch_size': args.gpt_batch_size,
              'num_workers': args.num_workers,
              'prefetch_factor': args.prefetch_factor,
              'collate_fn': dataset.collate,
              'shuffle': True,
              'drop_last': True}
    data_loader = torch.utils.data.DataLoader(dataset, **params)

    # hyperparameters of XLNet architecture
    mconf = GPTConfig(dataset.vocab_size, dataset.block_size, step_num=dataset.step_num, sample_num=dataset.total_sample_num,
                embd_pdrop=args.gpt_dropout, resid_pdrop=args.gpt_dropout, attn_pdrop=args.gpt_dropout,
                n_layer=args.gpt_layers, n_head=args.gpt_heads, n_embd=args.gpt_hidden_dim*args.gpt_heads, n_class=args.label_size)
    model = XLNet(mconf)

    # hyperparameters of XLNet training
    tokens_per_epoch = dataset.block_size * dataset.short_seq_num * len(dataset)
    final_tokens =  tokens_per_epoch * args.gpt_epochs
    tconf = TrainerConfig(batch_size=args.gpt_batch_size, block_size=dataset.block_size, short_seq_num=dataset.short_seq_num,
                        max_epochs=args.gpt_epochs, learning_rate=args.gpt_lr, betas = (0.9, 0.95), weight_decay=args.gpt_weight_decay,
                        lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=final_tokens,
                        ckpt_path='generator/gpt/save/{}_{}.pt'.format(args.gpt_train_name, split))

    start_time = perf_counter()
    trainer = Trainer(args, tconf, model, data_loader)
    trainer.train()
    print("[GPT] name: {}, split: {}, train time: {:.3f}".format(args.gpt_train_name, split, perf_counter() - start_time))

    return model


def generate(args, model, labels, ids, split):
    """
    Generate cluster ids for each node using CGT
    Args:
        args: arguments
        model: CGT model
        labels: label of each nodes
        ids: node id list
        split: split name ('train', 'val', 'test')
    """
    start_time = perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load('generator/gpt/save/{}_{}.pt'.format(args.gpt_train_name, split))
    model.load_state_dict(checkpoint)
    model.eval()

    n_samples = len(labels)
    start_id = args.cluster_num + 1
    start_ids = [start_id for _ in range(n_samples)]
    start_ids = torch.LongTensor(start_ids).unsqueeze(1).to(device)
    labels = torch.LongTensor(labels).to(device)

    result = []
    for b in range(int(math.ceil(n_samples / args.gpt_batch_size))):
        ind_start = b * args.gpt_batch_size
        ind_end = min(ind_start + args.gpt_batch_size, n_samples)
        generated_ids = sample(model, start_ids[ind_start:ind_end].contiguous(), labels[ind_start:ind_end].contiguous(), temperature=args.gpt_softmax_temperature)
        result.append(generated_ids[:, 1:])
    result = torch.cat(result, dim = 0)
    print("[GPT] name: {}, split: {}, generation time: {:.3f}".format(args.gpt_train_name, split, perf_counter() - start_time))

    return result.cpu()


def run(args, graphs, feats, labels, ids):
    """
    Learn graph distribution and generate new graphs using CGT
    Args:
        args: arguments
        graphs: a list of computation graphs
        feats: a list of feature matrices
        labels: a list of labels
        ids: a list of node ids
    """
    # Create save directory
    save_dir = 'generator/gpt/save'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # STEP 1: quantize features using cluster ids they are belonging to
    cluster_ids, cluster_centers = cluster_feats(args, feats)

    # STEP 2-1: train GPT on the original (training + validation) set
    target_ids = ids["train"] + ids["val"]
    model = train(args, graphs, cluster_ids, labels, target_ids, split="train")
    # STEP 2-2: generate cluster_ids for each (training + validation) set
    gen_train_ids = generate(args, model, labels[ids["train"]], ids["train"], split="train")
    gen_val_ids = generate(args, model, labels[ids["val"]], ids["val"], split="train")
    # STEP 2-3: creat dataset that map cluster ids to feature vectors
    train_dataset = QuantizedDataset(args, gen_train_ids, labels[ids["train"]], cluster_centers)
    val_dataset = QuantizedDataset(args, gen_val_ids, labels[ids["val"]], cluster_centers)

    # STEP 3-1: train GPT on the original test set
    target_ids = ids["test"]
    model = train(args, graphs, cluster_ids, labels, target_ids, split="test")
    # STEP 3-2: generate cluster_ids for the test set
    gen_test_ids = generate(args, model, labels[ids["test"]], ids["test"], split="test")
    # STEP 3-3: creat dataset that map cluster ids to feature vectors
    test_dataset = QuantizedDataset(args, gen_test_ids, labels[ids["test"]], cluster_centers)

    return train_dataset, val_dataset, test_dataset



