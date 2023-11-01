import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
from tqdm import tqdm
from time import perf_counter, sleep

from ..utils.utils import calc_f1, calc_loss

# GCN template
class GCN(nn.Module):
    """
    Graph Convolutional Networks (GCN)
    Args:
        model_name: model name ('gcn', 'gin', 'sgc', 'gat')
        input_dim: input dimension
        output_dim: output dimension
        hidden_dim: hidden dimension
        step_num: number of propagation steps
        output_layer: whether to use the output layer
    """

    def __init__(self, model_name, input_dim, output_dim, hidden_dim, step_num, output_layer=True):
        super(GCN, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.step_num = step_num

        self.output_layer = output_layer

        self.W =  nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False)])
        for _ in range(step_num-1):
            self.W.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        for w in self.W:
            nn.init.xavier_uniform_(w.weight)
        self.outputW = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.outputW.weight)

        self.pooling = 'sum' if model_name in ('gin', 'gat') else 'avg'
        self.nonlinear = (model_name != 'sgc')
        self.attention = (model_name == 'gat')
        if self.attention:
            self.attentionW = nn.Parameter(torch.empty(size=(2*hidden_dim, 1)))
            nn.init.xavier_uniform_(self.attentionW.data)
            self.leakyReLU = nn.LeakyReLU(0.2)
            self.softmax = nn.Softmax(dim=1)

    def get_parameters(self):
        ml = list()
        for w in self.W:
            ml.append({'params': w.parameters()})
        ml.append({'params': self.outputW.parameters()})
        if self.attention:
            ml.append({'params': self.attentionW})
        return ml

    def forward(self, feat, raw_adj):
        """
        Args:
            feat: feature matrix (batch_num * node_num, input_dim)
            raw_adj: adjacency matrix (batch_num, node_num, node_num)
        Returns:
            output: feature matrix of target nodes (batch_num, output_dim)
        """
        batch_num, batch_size = raw_adj.shape[0], raw_adj.shape[1]
        ids = torch.range(0, (batch_num - 1) * batch_size, batch_size, dtype=torch.long).to(raw_adj.device)
        adj = torch.block_diag(*raw_adj).to(raw_adj.device)
        if self.pooling == 'avg':
            adj = self.avg_pooling(feat, adj)
        X = feat
        for w in self.W:
            Z = w(X)
            if self.attention:
                adj = self.compute_attention(adj, Z)
            X = torch.spmm(adj, Z)
            if self.nonlinear:
                X = F.relu(X)
        if self.output_layer:
            X = self.outputW(X)
        return X[ids]

    def avg_pooling(self, feats, adj):
        """
        Args:
            feats: feature matrix (batch_num * node_num, input_dim)
            adj: adjacency matrix (batch_num * node_num, batch_num * node_num)
        Returns:
            adj: adjacency matrix (batch_num * node_num, batch_num * node_num)
        """
        nonzeros = torch.nonzero(torch.norm(feats, dim=1), as_tuple=True)[0]
        nonzero_adj = adj[:, nonzeros]
        row_sum = torch.sum(nonzero_adj, dim=1)
        row_sum = row_sum.masked_fill_(row_sum == 0, 1.)
        row_sum = torch.diag(1/row_sum).to(adj.device)
        adj = torch.spmm(row_sum, adj)
        return adj

    def compute_attention(self, adj, X):
        Wh1 = torch.matmul(X, self.attentionW[:self.hidden_dim, :])
        Wh2 = torch.matmul(X, self.attentionW[self.hidden_dim:, :])
        e = Wh1 + Wh2.T
        e= self.leakyReLU(e)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) #torch.tensor(-9e15).to(self.device))
        attention = F.softmax(attention, dim=1)
        return attention


def run(args, model_name, train_loader, val_loader, test_loader):
    """
    Evaluate GNN performance
    Args:
        args: arguments
        model_name: model name ('gcn', 'gin', 'sgc', 'gat')
        train_loader: training data loader
        val_loader: validation data loader
        test_loader: test data loader
    Returns:
        acc_mic: micro-F1 score
        acc_mac: macro-F1 score
    """

    device = args.device
    model = GCN(model_name, args.feat_size, args.label_size, args.hidden_dim, args.step_num)
    model = nn.DataParallel(model).to(device)

    # Test GCN models
    def test_model(args, model, data_loader, split='val'):
        start_time = perf_counter()
        stack_output = []
        stack_label = []
        model.eval()
        with tqdm(data_loader, unit="batch") as t_data_loader:
            for batch in t_data_loader:
                feats, adjs, labels = batch["feat"].to(device), batch["adj"].to(device), batch["label"].to(device)
                outputs = model(feats, adjs)
                loss = calc_loss(outputs, labels)
                stack_output.append(outputs.detach().cpu())
                stack_label.append(labels.cpu())
                t_data_loader.set_description(f"{split}")
                t_data_loader.set_postfix(loss=loss.item())
                sleep(0.1)
        stack_output = torch.cat(stack_output, dim=0)
        stack_label = torch.cat(stack_label, dim=0)
        loss = calc_loss(stack_output, stack_label)
        acc_mic, acc_mac = calc_f1(stack_output, stack_label)
        return loss, acc_mic, acc_mac

    ml = list()
    ml.extend(model.module.get_parameters())
    optimizer = optim.Adam(ml, lr=args.lr)

    patient = 0
    min_loss = np.inf
    for epoch in range(args.epochs):
        with tqdm(train_loader, unit="batch") as t_train_loader:
            for batch in t_train_loader:
                feats, adjs, labels = batch["feat"].to(device), batch["adj"].to(device), batch["label"].to(device)

                model.train()
                optimizer.zero_grad()
                outputs = model(feats, adjs)
                loss = calc_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                t_train_loader.set_description(f"Epoch {epoch}")
                t_train_loader.set_postfix(loss=loss.item())
                sleep(0.1)

        with torch.no_grad():
            new_loss, acc_mic, acc_mac = test_model(args, model, val_loader, 'val')
            if new_loss >= min_loss:
                patient = patient + 1
            else:
                min_loss = new_loss
                patient = 0

        if patient == args.early_stopping:
            break

    _, acc_mic, acc_mac = test_model(args, model, test_loader, 'test')

    del model
    return acc_mic, acc_mac

