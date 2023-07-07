import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import math
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
        h = F.normalize(h, p=2, dim=1)  # 行归一化
        return h


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, features, centers):
        return torch.sum(features * centers, dim=1)


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, block, h):
        # h：所有节点的mlp_h
        with block.local_scope():
            h_src = h
            h_dst = h[:block.number_of_dst_nodes()]
            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst

            block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            # 只返回dst_data的mean_h
            return block.dstdata['neigh']


class Encoder(nn.Module):
    def __init__(self, in_dim,  out_dim, activation):
        super().__init__()
        self.encoder = MLP(in_dim, out_dim, activation)
        self.meanAgg = MeanAggregator()

        
    def forward(self, block, h):
        # h:所有节点的raw_feat
        h = self.encoder(h)
        mean_h = self.meanAgg(block, h)
        # 只返回dst_node的h和mean_h
        return h[:block.number_of_dst_nodes()], mean_h


class LocalModel(nn.Module):
    # local inconsistency实现
    def __init__(self, in_dim, out_dim, activation) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim, out_dim, activation)
        self.discriminator = Discriminator()
        self.loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.MSELoss()
    
    def forward(self, block, h, out_nodes):
        # 前一个h：block.dst_node的feat
        # 后一个h：当前block里所有节点的feat
        h, mean_h = self.encoder(block, h)
        
        # positive
        pos = self.discriminator(h, mean_h)
        # negtive
        neg_idx = idx_sample(out_nodes).cuda()
        neg_neigh_h = mean_h[neg_idx]
        neg = self.discriminator(h, neg_neigh_h)

        l1 = self.loss(pos, torch.ones_like(pos))
        l2 = self.loss(neg, torch.zeros_like(neg))

        return l1 + l2, l1, l2, pos.detach()


# 向中心聚集
class GlobalModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation, nor_idx, ano_idx, center, labels, pos_diff):
        super().__init__()
        self.discriminator = Discriminator()
        self.beta = 0.9
        self.lamda = 0.6  # 两种损失占比
        self.neigh_weight = 0.2 # 邻居特征传播的平衡系数
        self.loss = nn.BCEWithLogitsLoss()
        self.nor_idx = nor_idx
        self.ano_idx = ano_idx
        self.center = center # hid_dim，由normal节点求得的固定的center
        self.encoder = Encoder(in_dim, out_dim, activation)
        self.pos_diff = pos_diff
        self.labels = labels
        self.pre_attn = self.pre_attention()

    def pre_attention(self):
        # 根据local inconsistency的相似度求出初始的attention
        nor_diff_mean = self.pos_diff[self.nor_idx].mean()
        nor_diff_std = torch.sqrt(self.pos_diff[self.nor_idx].var())


        normalized_pos = (self.pos_diff - nor_diff_mean) / nor_diff_std
        
        attn = 1-torch.sigmoid(normalized_pos)

        return attn.unsqueeze(1)

    def post_attention(self, h, mean_h):
        # 根据h和mean_h的相似度求出后续的attention
        simi = self.discriminator(h, mean_h)
        return simi.unsqueeze(1)

    def msg_pass(self, h, mean_h, attn):
        # h+attn*mean_h
        nei = attn * self.neigh_weight
        h = nei*mean_h + (1-nei)*h
        return h

    def forward(self, block, feats, out_nodes, epoch):
        h, mean_h = self.encoder(block, feats)

        pre_attn = self.pre_attn[out_nodes]

        post_attn = self.post_attention(h, mean_h)

        beta = math.pow(self.beta, epoch)
        if beta < 0.1:
            beta = 0.
        attn = beta*pre_attn + (1-beta)*post_attn

        h = self.msg_pass(h, mean_h, attn)

        scores = self.discriminator(h, self.center)
        
        curr_mask = self.labels[out_nodes].bool()

        pos_center_simi = scores[~curr_mask]
        neg_center_simi = scores[curr_mask]
        
        pos_center_loss = self.loss(pos_center_simi, torch.ones_like(pos_center_simi, dtype=torch.float32))
        neg_center_loss = self.loss(neg_center_simi, torch.zeros_like(neg_center_simi, dtype=torch.float32))

        center_loss = pos_center_loss + neg_center_loss

        return center_loss, scores
