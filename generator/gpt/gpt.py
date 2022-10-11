from collections import defaultdict
import os
from jinja2 import UndefinedError
import torch

from time import perf_counter

from generator.gpt.dataset import DatasetType
from generator.gpt.model import GPT, GPTConfig, XLNet
from generator.gpt.trainer import Trainer, TrainerConfig
from generator.gpt.utils import sample
from generator.cluster import cluster_feat_list, map_back_to_features
from task.utils.sample import make_batch
from tqdm import tqdm
import math

train_ratio = 0.8

def train(args, adjs, cluster_ids, labels, ids, train_name, split_name='train'):
    # Hyper parameters
    step_num = args.subgraph_step_num
    sample_num = args.subgraph_sample_num + args.noise_num

    # Define sequences
    start_time = perf_counter()
    """
    train_ids = ids[:int(train_ratio * len(ids))]
    val_ids = ids[int(train_ratio * len(ids)):]
    train_dataset = DatasetType(args, adjs, cluster_ids, labels, train_ids)
    val_dataset = DatasetType(args, adjs, cluster_ids, labels, val_ids)
    comp_ids = torch.cat((train_dataset.comp_ids(), val_dataset.comp_ids()), dim = 0)
    """
    train_dataset = DatasetType(args, adjs, cluster_ids, labels, ids)
    val_dataset = None
    comp_ids = train_dataset.comp_ids()
    print("[GPT] data preparation time: {:.3f}".format(perf_counter() - start_time))

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, step_num=step_num, sample_num=sample_num,
                embd_pdrop=args.gpt_dropout, resid_pdrop=args.gpt_dropout, attn_pdrop=args.gpt_dropout,
                n_layer=args.gpt_layers, n_head=args.gpt_heads, n_embd=args.gpt_hidden_dim*args.gpt_heads, n_class=args.label_size)
    if args.gpt_model == "GPT":
        model = GPT(mconf)
    elif args.gpt_model == "XLNet":
        model = XLNet(mconf)
    else:
        raise(UndefinedError)

    tokens_per_epoch = train_dataset.block_size * train_dataset.short_seq_num * len(train_dataset)
    final_tokens =  tokens_per_epoch * args.gpt_epochs
    tconf = TrainerConfig(max_epochs=args.gpt_epochs, batch_size=args.gpt_batch_size, learning_rate=args.gpt_lr,
                        betas = (0.9, 0.95), weight_decay=args.gpt_weight_decay, lr_decay=True,
                        warmup_tokens=tokens_per_epoch, final_tokens=final_tokens,
                        ckpt_path='generator/gpt/save/model_{}_{}.pt'.format(train_name, split_name),
                        num_workers=4)

    start_time = perf_counter()
    trainer = Trainer(model, train_dataset, val_dataset, tconf, args)
    trainer.train(split_name)
    print("[GPT] train name: {}, split: {}".format(train_name, split_name))
    print("[GPT] train time: {:.3f}".format(perf_counter() - start_time))

    return train_name, model, comp_ids

def generate(args, model, label_list, train_name, split_name):
    start_time = perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load('generator/gpt/save/model_{}_{}.pt'.format(train_name, split_name))
    model.load_state_dict(checkpoint)
    model.eval()

    step_num = args.subgraph_step_num
    sample_num = args.subgraph_sample_num + args.noise_num

    n_samples = len(label_list)
    start_id = args.cluster_num + 1
    start_ids = [start_id for _ in range(n_samples)]
    start_ids = torch.LongTensor(start_ids).unsqueeze(1).to(device)
    label_list = torch.LongTensor(label_list).to(device)

    result = []
    for b in range(int(math.ceil(n_samples/args.gpt_batch_size))):
        ind_start = b * args.gpt_batch_size
        ind_end = min(ind_start + args.gpt_batch_size, n_samples)
        generated_ids = sample(model, start_ids[ind_start:ind_end].contiguous(), \
                               label_list[ind_start:ind_end].contiguous(), temperature=args.gpt_softmax_temperature)
        result.append(generated_ids[:, 1:])
    result = torch.cat(result, dim = 0)
    print("[GPT] generation time: {:.3f}".format(perf_counter() - start_time))

    return result


def run(args, graphs, feats, labels, ids, train_name='default'):
    # label_list: list of labels (length = number of nodes)
    # Create save directory
    save_dir = 'generator/gpt/save'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    generated_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Cluster features and discretize features using cluster ids they are belonging to
    cluster_ids, cluster_center = cluster_feat_list(args, feats, train_name)
    for split in ids.keys():
        split_ids = ids[split]

        # Train GPT on cluster ids and generate synthetic cluster ids
        model_name, model, comp_ids = train(args, graphs, cluster_ids, labels, split_ids, train_name, split_name=split)
        generated_ids = generate(args, model, labels[split_ids], train_name, split_name=split)

        # Map cluster ids to feature vectors
        feat, graph, label, label_id = make_batch(args, comp_ids, labels[split_ids], cluster_center)
        generated_list['cluster']['feats'][split] = feat
        generated_list['cluster']['graphs'][split] = graph
        generated_list['cluster']['labels'][split] = label
        generated_list['cluster']['ids'][split] = label_id

        feat, graph, label, label_id = make_batch(args, generated_ids, labels[split_ids], cluster_center)
        generated_list['generate']['feats'][split] = feat
        generated_list['generate']['graphs'][split] = graph
        generated_list['generate']['labels'][split] = label
        generated_list['generate']['ids'][split] = label_id

    return generated_list, cluster_center



