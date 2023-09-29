import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import math
import numpy as np
from utils import idx_sample, row_normalization


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, activation) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            activation,
        ])  
    
    def forward(self, features):
        h = features
        for layer in self.encoder:
            h = layer(h)
        h = F.normalize(h, p=2, dim=1) 
        return h


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            return graph.ndata['neigh']


class Discriminator(nn.Module):
    def __init__(self, hid_dim) -> None:
        super().__init__()
    
    def forward(self, features, centers):
        return torch.sum(features * centers, dim=1)


class Encoder(nn.Module):
    def __init__(self, graph, in_dim,  out_dim, activation):
        super().__init__()
        self.encoder = MLP(in_dim, out_dim, activation)

        self.meanAgg = MeanAggregator()
        self.g = graph
        
    def forward(self, h):
        h = self.encoder(h)
        mean_h = self.meanAgg(self.g ,h)

        return h, mean_h


class LocalModel(nn.Module):
    def __init__(self, graph, in_dim, out_dim, activation, ano_idx) -> None:
        super().__init__()
        self.encoder = Encoder(graph, in_dim, out_dim, activation)
        self.g = graph
        self.discriminator = Discriminator(out_dim)
        self.loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()
        self.labeled_ano = ano_idx
        self.unlabeled_idx = self.unlabeled_idx_solve()

    def unlabeled_idx_solve(self):
        num_nodes = self.g.num_nodes()
        node_idx = np.arange(0, num_nodes)
        unlabeled_mask = np.ones(num_nodes, dtype=bool)
        unlabeled_mask[self.labeled_ano] = False

        return node_idx[unlabeled_mask]
    
    def forward(self, h):
        h, mean_h = self.encoder(h)
        
        neg_labeled_ano = self.discriminator(h[self.labeled_ano], mean_h[self.labeled_ano])

        pos = self.discriminator(h[self.unlabeled_idx], mean_h[self.unlabeled_idx])

        idx = torch.arange(0, h.shape[0])
        neg_idx = idx_sample(idx)
        neg_neigh_h = mean_h[neg_idx]
        unlabeled_neg = self.discriminator(h[self.unlabeled_idx], neg_neigh_h[self.unlabeled_idx])
        neg = torch.cat((neg_labeled_ano, unlabeled_neg))

        tmp_pos = torch.empty(h.shape[0]).cuda()
        tmp_pos[self.unlabeled_idx] = pos
        tmp_pos[self.labeled_ano] = neg_labeled_ano

        self.g.ndata['pos'] = tmp_pos

        l1 = self.loss(pos, torch.ones_like(pos))
        l2 = self.loss(neg, torch.zeros_like(neg))

        return l1 + l2, l1, l2


class GlobalModel(nn.Module):
    def __init__(self, graph, in_dim, out_dim, activation, nor_idx, ano_idx, center):
        super().__init__()
        self.g = graph
        self.discriminator = Discriminator(out_dim)
        self.beta = 0.9
        self.neigh_weight = 0.2
        self.loss = nn.BCEWithLogitsLoss()
        self.nor_idx = nor_idx
        self.ano_idx = ano_idx
        self.center = center
        self.encoder = Encoder(graph, in_dim, out_dim, activation)
        self.pre_attn = self.pre_attention()

    def pre_attention(self):
        msg_func = lambda edges:{'abs_diff': torch.abs(edges.src['pos'] - edges.dst['pos'])}
        red_func = lambda nodes:{'pos_diff': torch.mean(nodes.mailbox['abs_diff'], dim=1)}
        self.g.update_all(msg_func, red_func)

        pos = self.g.ndata['pos']
        pos.requires_grad = False

        pos_diff = self.g.ndata['pos_diff'].detach()

        diff_mean = pos_diff[self.nor_idx].mean()
        diff_std = torch.sqrt(pos_diff[self.nor_idx].var())

        normalized_pos = (pos_diff - diff_mean) / diff_std
        
        attn = 1-torch.sigmoid(normalized_pos)

        return attn.unsqueeze(1)

    def post_attention(self, h, mean_h):
        simi = self.discriminator(h, mean_h)
        return simi.unsqueeze(1)


    def msg_pass(self, h, mean_h, attn):
        # h+attn*mean_h
        nei = attn * self.neigh_weight
        h = nei*mean_h + (1-nei)*h
        return h

    def forward(self, feats, epoch):
        h, mean_h = self.encoder(feats)

        post_attn = self.post_attention(h, mean_h)
        beta = math.pow(self.beta, epoch)
        if beta < 0.1:
            beta = 0.
        attn = beta*self.pre_attn + (1-beta)*post_attn

        h = self.msg_pass(h, mean_h, attn)

        scores = self.discriminator(h, self.center)
        
        pos_center_simi = scores[self.nor_idx]
        neg_center_simi = scores[self.ano_idx]
        
        pos_center_loss = self.loss(pos_center_simi, torch.ones_like(pos_center_simi, dtype=torch.float32))
        neg_center_loss = self.loss(neg_center_simi, torch.zeros_like(neg_center_simi, dtype=torch.float32))

        center_loss = pos_center_loss + neg_center_loss

        return center_loss, scores
