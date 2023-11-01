# Graph Generative Model for Benchmarking Graph Neural Networks

We propose a novel, modern graph generation problem to enable generating privacy-controlled, synthetic substitutes of large-scale real-world graphs that can be effectively used to evaluate GNN models.
Our proposed graph generative model, Computation Graph Transformer (CGT) 1) operates on minibatches rather than the whole graph, avoiding scalability issues, and 2) reduces the task of learning graph distributions to learning feature vector sequence distributions, which we approach with a novel Transformer architecture.

You can see our [ICML 2023 paper](https://arxiv.org/abs/2207.04396) for more details.

## Setup
Create a new conda environment, install [PyTorch](https://pytorch.org) and the remaining requirements:
```
conda create python==3.7 -n cgt
conda activate cgt
pip install -r requirement.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
The code is implemented on PyTorch DataParallel.

## Dataset
You can download public graph datasets in the npz format from [GNN-Benchmark](https://github.com/shchur/gnn-benchmark).
Place the dataset in `data/` directory.
For your convenience, `cora.npz` and `citeseer.npz` are already saved in `data\`.
We also support ogbn-arxiv and ogbn-products dataset from [OGBN benchmark](https://ogb.stanford.edu/docs/nodeprop/). 

## Usage
In `run.sh`, you write down a list of graph datasets that you want to learn distributions into `DATASETS`.
First, we add different sizes of noisy neighbors to augment the original graphs using `NOISES`.
By executing `run.sh`, we learn three different distributions with different noise sizes `NOISES=(0 2 4)` for each dataset.
For each dataset, we train three different GCN models (GCN, GIN, SGC) on a pair of original and synthetic graphs, and then compare their performance.
The details of other hyperparameters can be found in args.py.

## Differential Privacy module

As described in the main paper, DP-SGD on transformer performs badly. 
Thus we provide only DP-k-means module in this repository.
To run DP-k-means, you need to download an open-source library from: https://github.com/google/differential-privacy/tree/main/learning/clustering
Then you can uncomment line 11-12 in `generator/cluster.py` and set `dp_feature` in `args.py` to True.


## File description

We provide brief descriptions for each file as follows:

| Directory/File | description |
| ---- | ---- |
| run.sh | script to run experiments |
| args.py | set hyperparameters |
| test.py | main file: prepare models, read datasets, graph generation, GNN evaluation |
| data/ | download datasets |
| generator/ | codes related to graph transformer |
| generator/cluster.py | k-means or DP k-means clustering |
| generator/gpt | CGT main directory |
| generator/gpt/gpt.py | prepare models, prepare datasets, train/generation loops |
| generator/gpt/dataset.py | dataset for flatten computation graphs |
| generator/gpt/model.py | XLNet model |
| generator/gpt/trainer.py | training loop |
| generator/gpt/utils.py | generation loop |
| task/ | GNN models |
| task/aggregation | GNN models with different aggregation strategies (GCN, GAT, SGN, GIN) |
| task/utils/dataset.py | Computation Graph Dataset for PyTorch DataParallel |
| task/utils/utils.py | ogbn/npz format datasets loading, utility functions |


### Citation
Please consider citing the following paper when using our code for your application.

```
@article{yoon2022scalable,
  title={Scalable Privacy-enhanced Benchmark Graph Generative Model for Graph Convolutional Networks},
  author={Yoon, Minji and Wu, Yue and Palowitch, John and Perozzi, Bryan and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2207.04396},
  year={2022}
}
```
