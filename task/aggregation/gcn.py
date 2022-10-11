import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
from time import perf_counter

from ..utils.utils import calc_f1, calc_loss

# GCN template
class GCN(nn.Module):

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

    def forward(self, feats, adj):
        if self.pooling == 'avg':
            adj = self.avg_pooling(feats, adj)
        X = feats
        for w in self.W:
            Z = w(X)
            if self.attention:
                adj = self.compute_attention(adj, Z)
            X = torch.spmm(adj, Z)
            if self.nonlinear:
                X = F.relu(X)
        if self.output_layer:
            X = self.outputW(X)
        return X

    def avg_pooling(self, feats, adj):
        nonzeros = torch.nonzero(torch.norm(feats, dim=1), as_tuple=True)[0]
        nonzero_adj = adj[:, nonzeros]
        row_sum = torch.sum(nonzero_adj, dim=1)
        row_sum = row_sum.masked_fill_(row_sum == 0, 1.)
        row_sum = torch.diag(1/row_sum).to(self.device)
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


def run(args, model_name, feats, graphs, labels, ids, feature_matrix=None):
    """
    Evaluate GNN performance
    """

    device = args.device
    model = GCN(model_name, args.feat_size, args.label_size, args.hidden_dim, args.step_num)
    model = model.to(device)

    # Test GCN models
    def test_model(args, model, feats, graphs, labels, ids):
        start_time = perf_counter()
        device = args.device
        stack_output = []
        stack_label = []
        model.eval()
        for feat, adj, label, id in zip(feats, graphs, labels, ids):
            if feature_matrix is not None:
                feat = feature_matrix[feat]
            feat, adj, label = feat.to(device), adj.to(device), label.to(device)
            output = model(feat, adj)
            stack_output.append(output[id].detach().cpu())
            stack_label.append(label.cpu())
        stack_output = torch.cat(stack_output, dim=0)
        stack_label = torch.cat(stack_label, dim=0)
        loss = calc_loss(stack_output, stack_label)
        acc_mic, acc_mac = calc_f1(stack_output, stack_label)
        test_time = perf_counter()-start_time
        return loss, acc_mic, acc_mac, test_time

    ml = list()
    ml.extend(model.get_parameters())
    optimizer = optim.Adam(ml, lr=args.lr)

    start_time = perf_counter()
    patient = 0
    min_loss = np.inf
    for epoch in range(args.epochs):
        for feat, adj, label, id in zip(feats['train'], graphs['train'], labels['train'], ids['train']):
            if feature_matrix is not None:
                feat = feature_matrix[feat]
            feat, adj, label = feat.to(device), adj.to(device), label.to(device)

            model.train()
            optimizer.zero_grad()
            output = model(feat, adj)
            loss = calc_loss(output[id], label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            new_loss, acc_mic, acc_mac, val_time = test_model(args, model, feats['val'], graphs['val'], labels['val'], ids['val'])
            if new_loss >= min_loss:
                patient = patient + 1
            else:
                min_loss = new_loss
                patient = 0

        if patient == args.early_stopping:
            break

    train_time = perf_counter() - start_time
    test_loss, acc_mic, acc_mac, test_time = test_model(args, model, feats['test'], graphs['test'], labels['test'], ids['test'])

    del model
    return acc_mic, acc_mac, train_time, test_time

