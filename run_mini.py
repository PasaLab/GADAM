import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import torch
import torch.nn as nn
import dgl
from dgl.data import register_data_args
import argparse, time
from model_mini import *
from utils import *
from sklearn.metrics import roc_auc_score


def train_local(net, graph, labels, dataloader, opt, args, init=True):
    memo = {}

    device = args.gpu
    if device >= 0:
        torch.cuda.set_device(device)
        net = net.to(device)

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    
    if init:
        net.apply(init_xavier)
    
    print('train on:', 'cpu' if device<0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    batchs = len(dataloader)

    pos_memo = torch.zeros(graph.num_nodes())

    for epoch in range(args.local_epochs):
        sum_loss = sum_l1 = sum_l2 = 0
        net.train()
        for in_nodes, out_nodes, blocks in dataloader:
            block = blocks[0]  
            block = block.to(device)  
            input_features = blocks[0].srcdata['feat']		
            loss, l1, l2, pos = net(block, input_features, out_nodes)

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_loss += loss.item()
            sum_l1 += l1.item()
            sum_l2 += l2.item()

        net.eval()
        for _, out_nodes, blocks in dataloader:
            block = blocks[0]  
            block = block.to(device)  
            input_feats = blocks[0].srcdata['feat']	
            
            _, _, _, pos = net(block, input_feats, out_nodes)
 
            pos_memo[out_nodes] = pos.cpu()

        auc = roc_auc_score(labels.numpy(), -pos_memo.numpy())

        mean_loss = sum_loss/batchs
        if mean_loss < best:
            best = mean_loss
            torch.save(net.state_dict(), 'best_local_model_mini.pkl')

        print("Epoch {} | Loss {:.4f} | l1 {:.4f} | l2 {:.4f} | auc {:.4f} "
               .format(epoch+1, mean_loss , sum_l1/batchs, sum_l2/batchs, auc))

    net.load_state_dict(torch.load('best_local_model_mini.pkl'))
    h_memo = torch.empty((graph.num_nodes(), args.out_dim))

    for _, out_nodes, blocks in dataloader:
        block = blocks[0]  
        block = block.to(device)  
        input_features = blocks[0].srcdata['feat']	
        
        h, _ = net.encoder(block, input_features)
        h_memo[out_nodes] = h.detach().cpu()

    memo['pos'] = pos_memo
    memo['h'] = h_memo

    torch.save(memo, 'memo.pth')


def load_info_from_local(graph, device):
    memo = torch.load('memo.pth')
    num_nodes = graph.num_nodes()
    h = memo['h']
    scores = memo['pos']
    ano_topk = 0.01  
    nor_topk = 0.3

    num_ano = int(num_nodes * ano_topk)
    _, ano_idx = torch.topk(scores, num_ano)

    num_nor = int(num_nodes * nor_topk)
    _, nor_idx = torch.topk(-scores, num_nor)

    center = h[nor_idx].mean(dim=0)

    if device >= 0:
        memo = {k: v.to(device) for k, v in memo.items()}
        nor_idx = nor_idx.cuda()
        ano_idx = ano_idx.cuda()
        center = center.cuda()
    
    graph.ndata['pos'] = scores.cuda()
    msg_func = lambda edges:{'abs_diff': torch.abs(edges.src['pos'] - edges.dst['pos'])}
    red_func = lambda nodes:{'pos_diff': torch.mean(nodes.mailbox['abs_diff'], dim=1)}
    graph.update_all(msg_func, red_func)
    pos_diff = graph.ndata['pos_diff']

    # nor_mean = pos_diff[nor_idx].mean()
    # nor_std = torch.sqrt(pos_diff[nor_idx].var())

    return memo, nor_idx, ano_idx, center, pos_diff


def train_global(global_net, dataloader, memo, opt, labels, num_nodes, args):
    epochs = args.global_epochs

    device = args.gpu
    pos = memo['pos']

    if device >= 0:
        torch.cuda.set_device(device)
        global_net = global_net.to(device)
        labels = labels.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    
    init = True
    if init:
        global_net.apply(init_xavier)
    
    print('train on:', 'cpu' if device<0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    dur = []
    batchs = len(dataloader)
    scores_memo = torch.zeros(num_nodes)

    for epoch in range(epochs):
        sum_loss = 0
        global_net.train()
        for _, out_nodes, blocks in dataloader:
            block = blocks[0] 
            block = block.to(device)  
            input_feats = blocks[0].srcdata['feat']	
            
            loss, scores = global_net(block, input_feats, out_nodes, epoch)
            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_loss += loss.item()

        # eval
        global_net.eval()
        for _, out_nodes, blocks in dataloader:
            block = blocks[0]  
            block = block.to(device)  
            input_feats = blocks[0].srcdata['feat']	
            
            _, scores = global_net(block, input_feats, out_nodes, epoch)
            scores_memo[out_nodes] = scores.detach().cpu()
        
        mix_score = (pos.cpu() + scores_memo) / 2 

        mix_auc = roc_auc_score(labels.cpu().numpy(), -mix_score)
        auc = roc_auc_score(labels.cpu().numpy(), -scores_memo)

        mean_loss = sum_loss / batchs

        if mean_loss < best:
            best = mean_loss
            torch.save(global_net.state_dict(), 'best_global_model_mini.pkl')

        print("Epoch {} | Loss {:.4f} | mix-auc {:.4f}".format(epoch+1, sum_loss/batchs, mix_auc))


def main(args):
    seed_everything(args.seed)
    
    graph = my_load_data(args.data)
    feats = graph.ndata['feat']
    labels = graph.ndata['label']
    num_nodes = graph.num_nodes()

    if args.gpu >= 0:
        graph = graph.to(args.gpu)
    
    batch_size = args.batch_size
    node_idx = torch.arange(graph.number_of_nodes(), device=args.gpu)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.DataLoader(
        graph, 
        node_idx,
        sampler,
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False)

    in_feats = feats.shape[1]

    local_net = LocalModel(in_feats, args.out_dim, nn.PReLU(),)

    local_opt = torch.optim.Adam(local_net.parameters(), 
                                 lr=args.local_lr, 
                                 weight_decay=args.weight_decay)

    # train_local(local_net, graph, labels, dataloader, local_opt, args)

    memo, nor_idx, ano_idx, center, pos_diff = load_info_from_local(graph, args.gpu)


    global_train_idx = torch.cat((nor_idx, ano_idx))
    global_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    global_dataloader = dgl.dataloading.DataLoader(
        graph, 
        global_train_idx,
        global_sampler,
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False) 

    global_net = GlobalModel(in_feats, 
                             args.out_dim, 
                             nn.PReLU(), 
                             nor_idx, 
                             ano_idx, 
                             center,
                             labels,
                             pos_diff)
    opt = torch.optim.Adam(global_net.parameters(), 
                                 lr=args.global_lr, 
                                 weight_decay=args.weight_decay)

    train_global(global_net, global_dataloader, memo, opt, labels, num_nodes, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    register_data_args(parser)
    parser.add_argument("--data", type=str, default="products",
                        help="dataset")
    parser.add_argument("--seed", type=int, default=124,
                        help="random seed")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--local-lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--global-lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--local-epochs", type=int, default=1,
                        help="number of training local model epochs")
    parser.add_argument("--global-epochs", type=int, default=20,
                        help="number of training global model epochs")
    parser.add_argument("--out-dim", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of hidden gcn units")
    parser.add_argument("--beta", type=float, default=1.,
                        help="attn_loss weight")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=True)

    args = parser.parse_args()
    print(args)
    main(args)
