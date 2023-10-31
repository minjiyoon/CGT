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

## DataSet
You can download public graph datasets in the npz format from [GNN-Benchmark](https://github.com/shchur/gnn-benchmark).
Place the dataset in `data/` directory.
For your convenience, `cora.npz` and `citeseer.npz` are already saved in `data\`.

## Usage
Execute `run.sh` to run 3 different GCN models with variable noisy edge numbers on the given graph.
The details of other optional hyperparameters can be found in args.py.

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
