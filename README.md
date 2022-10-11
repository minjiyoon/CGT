# SCALABLE AND PRIVACY-ENHANCED GRAPH GENERATIVE MODEL FOR GRAPH NEURAL NETWORKS

We propose a novel, modern graph generation problem to enable generating privacy-controlled, synthetic substitutes of large-scale real-world graphs that can be effectively used to evaluate GNN models.
More specifically, we propose a novel graph generative model, Computation Graph Transformer (CGT) that can learn and reproduce the distribution of real-world graphs in a privacy-enhanced way.

You can see our [preprint paper](https://arxiv.org/abs/2207.04396) for more details.

## Overview
`Data/` directory contains all files to preprocess OAG-CS raw datasets and extract OAG-ML and OAG-CN subgraphs.
`Model/` directory contains how to train HGNN and KTN models on the preprocessed heterogeneous datasets. 
  
## Setup
This implementation is based on python==3.7. To run the code, you need the dependencies listed in `requirement.txt'

## OAG DataSet
Our current experiments are conducted on Open Academic Graph on Computer Science field (OAG-CS). 
More information to how to download and preprocess OAG-CS dataset can be found in `Data/` directory.

## Usage
Execute `MODEL/run_oag.sh` to run 8 different zero-shot transfer learning tasks on the OAG-CS graph using KTN.
The details of other optional hyperparameters can be found in args.py.

### Citation
Please consider citing the following paper when using our code for your application.

```bibtex
@article{yoon2022zero,
  title={Zero-shot Domain Adaptation of Heterogeneous Graphs via Knowledge Transfer Networks},
  author={Yoon, Minji and Palowitch, John and Zelle, Dustin and Hu, Ziniu and Salakhutdinov, Ruslan and Perozzi, Bryan},
  journal={arXiv preprint arXiv:2203.02018},
  year={2022}
}
```
