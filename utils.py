import dgl
import torch
import torch.nn.functional as F
import random
import os
import dgl.function as fn
from dgl.data.utils import load_graphs
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def t_v_t_split(train_ratio, val_ratio, num_nodes):
    node_idx = np.arange(num_nodes)
    train_num = int(train_ratio * num_nodes)
    val_num = int(val_ratio * num_nodes)

    selected_idx = np.random.choice(node_idx, train_num+val_num, replace=False)

    train_mask = torch.zeros(num_nodes).bool()
    val_mask = torch.zeros(num_nodes).bool()

    train_mask[selected_idx[:train_num]] = True
    val_mask[selected_idx[train_num:]] = True
    test_mask = torch.logical_and(~train_mask, ~val_mask)

    print(torch.sum(train_mask))
    print(torch.sum(val_mask))
    print(torch.sum(test_mask))
    return train_mask, val_mask, test_mask


def idx_sample(idxes):
    num_idx = len(idxes)
    random_add = torch.randint_like(idxes, high=num_idx, device='cpu')
    idx = torch.arange(0, num_idx)

    shuffled_idx = torch.remainder(idx+random_add, num_idx)

    return shuffled_idx

def row_normalization(feats):
    return F.normalize(feats, p=2, dim=1)


def load_data(dataname, path='./raw_dataset/Flickr'):
    data = sio.loadmat(f'{path}/{dataname}.mat')

    adj = data['Network'].toarray()
    feats = torch.FloatTensor(data['Attributes'].toarray())
    label = torch.LongTensor(data['Label'].reshape(-1))

    graph = dgl.from_scipy(coo_matrix(adj)).remove_self_loop()
    graph.ndata['feat'] = feats
    graph.ndata['label'] = label

    return graph


def my_load_data(dataname, path='./data/'):
    data_dir = path+dataname+'.bin'
    graph = load_graphs(data_dir)

    return graph[0][0]


# def score_prop(g,  global_net, feats, epochs):
#     beta = 0.1
#     attn = g.ndata['attn'].detach().squeeze()
#     labels = g.ndata['label']
#     with g.local_scope():
#         g.ndata['new_score'] = scores.detach().clone()
#         for i in range(epochs):
#             new_score = g.ndata['new_score']
#             g.update_all(fn.copy_u('new_score', 'm'), fn.mean('m', 'nei_score'))
#             nei_score = g.ndata['nei_score']
#             new_score = (beta*attn)*nei_score + (1-beta*attn)*new_score
#             new_score = new_score.detach()
#             g.ndata['new_score'] = new_score
#             auc = roc_auc_score(labels.numpy(), -new_score.numpy())
#             print(auc)