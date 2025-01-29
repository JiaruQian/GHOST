import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse,coalesce
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from utils import GW_utils

class Generator(nn.Module):
    def __init__(self, noise_dim, input_dim, output_dim, dropout):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(output_dim, output_dim)
        
        hid_layers = []
        dims = [noise_dim+output_dim, 64, 128, 256]
        for i in range(len(dims)-1):
            d_in = dims[i]
            d_out = dims[i+1]
            hid_layers.append(nn.Linear(d_in, d_out))
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p=dropout, inplace=False))
        self.hid_layers = nn.Sequential(*hid_layers)
        self.nodes_layer = nn.Linear(256, input_dim)

    def forward(self, z, c):
        z_c = torch.cat((self.emb_layer.forward(c), z), dim=-1)
        hid = self.hid_layers(z_c)
        node_logits = self.nodes_layer(hid)
        return node_logits
    

def construct_graph(node_logits, adj_logits, k=5):

    adjacency_matrix = torch.zeros_like(adj_logits)

    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)

    for i in range(node_logits.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge = adjacency_matrix.long()
    edge_index, _ = dense_to_sparse(edge)
    edge_index = add_self_loops(edge_index)[0]
    data = Data(x=node_logits, edge_index=edge_index)
    
    # edge_index = coalesce(edge_index, None, node_logits.size(0), node_logits.size(0))[0]

    # data = Data(x=node_logits, edge_index=edge_index)
    return data  


def feature_loss(pred_X, target_X, lambda_d):

    pred_X_log = F.log_softmax(pred_X, dim=1)  
    target_X_prob = F.softmax(target_X, dim=1)

    
    div_loss = F.kl_div(pred_X_log, target_X_prob, reduction='batchmean')
    
    cosine_sim = F.cosine_similarity(pred_X, target_X, dim=-1)
    
    
    disp_loss = 1 - cosine_sim.mean()


    loss_feat =  lambda_d *div_loss +  disp_loss

    return loss_feat

def gw_distance_loss(A1,A2, X1, X2,device, alpha=0.5, learn_alpha=False):



    alpha_tensor = torch.tensor([alpha], dtype=torch.float32, device=device, requires_grad=learn_alpha)
    
    h1 = torch.ones(A1.size(0), dtype=torch.float32, device=device) / A1.size(0)
    h2 = torch.ones(A2.size(0), dtype=torch.float32, device=device) / A2.size(0)
    
    #cost matrix
    M = torch.cdist(X1, X2, p=2) ** 2  
    M = M.to(device)

    fgw_dist, gp, gq, gC1, gC2, gF1, gF2, galpha = GW_utils.parallel_fused_gromov_wasserstein2_learnablealpha(
        C1=A1, 
        C2=A2, 
        F1=X1, 
        F2=X2, 
        M=M, 
        p=h1, 
        q=h2, 
        alpha=alpha_tensor, 
        compute_gradients=True, 
        learn_alpha=learn_alpha
    )
    

    if learn_alpha:
        gw_loss = GW_utils.set_gradients(
            GW_utils.ValFunction, 
            fgw_dist, 
            (X1, X2, alpha_tensor), 
            (gF1, gF2, galpha)
        )
    else:
        gw_loss = GW_utils.set_gradients(
            GW_utils.ValFunction, 
            fgw_dist, 
            (X1, X2), 
            (gF1, gF2)
        )
    
    return gw_loss



def gen_y_noise(num_gen, num_classes,device):
    c_cnt = [0] * num_classes
    for class_i in range(num_classes):
        c_cnt[class_i] = int(num_gen * 1 / num_classes)
    c_cnt[-1] += num_gen - sum(c_cnt)

    y_noise = torch.zeros(num_gen).to(device).long()
    ptr = 0
    for class_i in range(num_classes):
        for _ in range(c_cnt[class_i]):
            y_noise[ptr] = class_i
            ptr += 1 

    shuffled_indices = torch.randperm(num_gen).to(device)
    y_noise = y_noise[shuffled_indices]
    return y_noise

def generate_random_one_hot_labels(num_classes, num_nodes):

    labels = torch.randint(0, num_classes, (num_nodes,))  
    
    one_hot_labels = F.one_hot(labels, num_classes=num_classes)  # [num_nodes, num_classes]
    
    return one_hot_labels


def compute_criterion(model, x, edge_index):

    h1 = F.relu(model.layers[0](x, edge_index))

    conv1_weight = model.layers[0].lin.weight

    topological_loss = 0.0
    for edge in edge_index.t():  
        i, j = edge[0].item(), edge[1].item()
        h_i = h1[i]
        h_j = h1[j]

        e_ij = (h_i @ conv1_weight).T @ (h_j @ conv1_weight).tanh()

        topological_loss += e_ij ** 2
        

    topological_loss.backward(retain_graph=True)


    criterion = {}
    criterion_one = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            criterion[name] = param.grad.clone()
            criterion_one[name] = param.grad.norm().item()
        
    return criterion , criterion_one