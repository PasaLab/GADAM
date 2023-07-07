import dgl
import numpy as np
import torch
from dgl.data.utils import save_graphs,load_graphs
from utils import *
from ogb.nodeproppred import DglNodePropPredDataset

def make_struct_ano(graph, ano_idx, clique):
    num_nodes = graph.num_nodes()
    rounds = int(len(ano_idx) / clique)
    clique_id = 1
    clique_label = torch.zeros(num_nodes)

    src = []
    tgt = []
    for i in range(rounds):
        print("rounds ", i)
        curr_idx = ano_idx[i*clique : (i+1)*clique]
        clique_label[curr_idx] = clique_id
        for j in curr_idx:
            for k in curr_idx:
                if j!=k and not graph.has_edges_between(j, k):
                    src.append(j)
                    tgt.append(k)
        clique_id += 1

    graph = dgl.add_edges(graph, src, tgt)
    graph.ndata['clique_id'] = clique_label
    return graph


def make_feat_ano(graph, ano_idx, around):
    feats = graph.ndata['feat']
    feat_aug = feats.clone()
    node_idx = np.arange(0, graph.num_nodes())
    cnt = 1
    for idx in ano_idx:
        print(cnt)
        candidate = np.random.choice(node_idx, around, replace=False)
        dis = torch.sqrt(torch.sum(torch.pow(feats[idx] - feats[candidate], 2), dim = 1))
        selected_idx = torch.argmax(dis)
        feat_aug[idx] = feats[candidate[selected_idx]]
        cnt += 1

    graph.ndata['feat'] = feat_aug


def make_full_ano(graph, ano_num, clique_size=15, ano_around=50):
    num_nodes = graph.num_nodes()

    node_idx = np.arange(0, num_nodes)
    selected_idx = np.random.choice(node_idx, ano_num, replace=False)

    num_struct_ano = int(ano_num/2)

    feat_label = torch.zeros(num_nodes)
    feat_label[selected_idx[:num_struct_ano]] = 1

    struct_label = torch.zeros(num_nodes)
    struct_label[selected_idx[num_struct_ano:]] = 1

    labels = torch.zeros(num_nodes)
    labels[selected_idx] = 1

    make_feat_ano(graph, selected_idx[:num_struct_ano], ano_around)
    graph = make_struct_ano(graph, selected_idx[num_struct_ano:], clique_size)
    graph.ndata['label'] = labels
    graph.ndata['feat_label'] = feat_label
    graph.ndata['struct_label'] = struct_label

    return graph

seed_everything(124)

# ACM、BlogCatalog
# graph = load_data('Flickr')

# Cora、Citeseer、Pubmed
# data = dgl.data.PubmedGraphDataset()
# graph = data[0]

# ogb-arxiv
dataset = DglNodePropPredDataset(name='ogbn-products')
graph = dataset[0][0]

path = './data/products.bin'

edge = graph.num_edges()

print(graph)

graph = make_full_ano(graph, 90000)

save_graphs(path, graph)

print("*"*10)

result = load_graphs(path)[0][0]

edge1 = result.num_edges()

struct_label = result.ndata['struct_label']
feat_label = result.ndata['feat_label']
print(torch.sum(struct_label))
print(torch.sum(feat_label))

print(result)

print('add edges: ',edge1 - edge)