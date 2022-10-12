# Scalable and Privacy-enhanced Graph Generative Model for Graph Neural Networks

We propose a novel, modern graph generation problem to enable generating privacy-controlled, synthetic substitutes of large-scale real-world graphs that can be effectively used to evaluate GNN models.
Our proposed graph generative model, Computation Graph Transformer (CGT) 1) operates on minibatches rather than the whole graph, avoiding scalability issues, and 2) reduces the task of learning graph distributions to learning feature vector sequence distributions, which we approach with a novel Transformer architecture.

You can see our [preprint paper](https://arxiv.org/abs/2207.04396) for more details.

## Overview
`Data/` directory contains all files to preprocess OAG-CS raw datasets and extract OAG-ML and OAG-CN subgraphs.
`Model/` directory contains how to train HGNN and KTN models on the preprocessed heterogeneous datasets. 
  
## Setup
This implementation is based on python==3.7. To run the code, you need the dependencies listed in `requirement.txt'

## DataSet
You can download public graph datasets in the npz format from [GNN-Benchmark](https://github.com/shchur/gnn-benchmark).
Place the dataset in `data/` directory.

## Usage
Execute `run_aggregation.sh` to run 4 different GCN models with variable noisy edge numbers on the given graph.
The details of other optional hyperparameters can be found in args.py.

### Citation
Please consider citing the following paper when using our code for your application.

```bibtex
@article{yoon2022scalable,
  title={Scalable Privacy-enhanced Benchmark Graph Generative Model for Graph Convolutional Networks},
  author={Yoon, Minji and Wu, Yue and Palowitch, John and Perozzi, Bryan and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2207.04396},
  year={2022}
}
```
