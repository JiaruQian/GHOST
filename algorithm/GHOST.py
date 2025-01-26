from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse,coalesce
from utils.utils import *

class GHOSTServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(GHOSTServer, self).__init__(args, clients, model, data, logger)
        self.args = args
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes= args.num_classes
        self.noise_dim = args.noise_dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=args.weight_decay)
        self.lambda_r = args.lambda_r
        self.lambda_n = args.lambda_n
        self.M = args.M
        self.T_G = args.T_G

    def aggregate(self):
        print('---------------------------')
        print('knowledge integration...')
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        criterion = None
        criterion_list = []
        old_params_list = []
        k = 0

        print('---------------------------')
        print('global model training...')
        for i, cid in enumerate(self.sampled_clients):

            num_gen = int(self.clients[cid].data.x.shape[0])
            generator = self.clients[cid].generator

            for _ in range(self.M):
                with torch.no_grad():
                    z_noise = torch.randn((num_gen, self.noise_dim), device=self.device).float()
                    y_noise = gen_y_noise(num_gen,self.num_classes,self.device)
                    node_logits = generator(z_noise,y_noise)
                    node_norm = F.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(
                        node_logits, adj_logits, k=5).to(self.device)

                
                criterion , criterion_one = compute_criterion(self.model,pseudo_graph.x, pseudo_graph.edge_index)
                criterion_list.append(criterion)

                for _ in range(self.T_G):
                    self.model.train()
                    self.optimizer.zero_grad()
                    self.model = self.model.to(self.device)
                    _,out = self.model(pseudo_graph)

                    # loss task
                    loss = F.nll_loss(out, y_noise)

                    
                    loss_retention = 0
                    loss_norm = 0
                    if k > 0:
                        for name, param in self.model.named_parameters():
                            for kk in range(k): 
                                if name in old_params_list[kk] and name in criterion_list[kk]: 
                                    loss_retention += (abs(criterion_list[kk][name]) * (param - old_params_list[kk][name]) ** 2).sum()
                            loss_norm += criterion_one.get(name, 0)
                            
                    loss = loss + self.lambda_r * loss_retention + self.lambda_n * loss_norm

                    loss.backward()
                    self.optimizer.step()
                
                
                old_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
                old_params_list.append(old_params)
                k = k + 1




class GHOSTClient(BaseClient):
    def __init__(self, args, model, data):
        super(GHOSTClient, self).__init__(args, model, data)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.feature_dim = self.data.x.shape[-1]
        self.num_classes= args.num_classes
        self.noise_dim = args.noise_dim
        self.generator = Generator(noise_dim=self.noise_dim,
                                   input_dim=self.feature_dim,
                                   output_dim=self.num_classes,
                                   dropout=0.2).to(self.device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lambda_d = args.lambda_d
        self.lambda_f = args.lambda_f


    def train(self):
        self.optimizer.zero_grad()
        num_gen = self.data.x.shape[0]

        self.z_noise = torch.randn((num_gen, self.noise_dim), device=self.device).float()

        node_logits = self.generator(self.z_noise,self.data.y)
        node_norm = F.normalize(node_logits, p=2, dim=1)
        adj_logits = torch.mm(node_norm, node_norm.t())
        pseudo_graph = construct_graph(
                        node_logits, adj_logits, k=5)


        
        A_ori = to_dense_adj(edge_index=self.data.edge_index,max_num_nodes=num_gen).squeeze(0)
        A_new = to_dense_adj(edge_index=pseudo_graph.edge_index,max_num_nodes=num_gen).squeeze(0)
        A_new = A_new // 2
        
        

        loss_feat = feature_loss(pseudo_graph.x[self.data.train_mask],self.data.x[self.data.train_mask],lambda_d = self.lambda_d)
        loss_fgw = gw_distance_loss(A_ori[self.data.train_mask][:, self.data.train_mask],
                                A_new[self.data.train_mask][:, self.data.train_mask],
                                self.data.x[self.data.train_mask],
                                pseudo_graph.x[self.data.train_mask],self.device)
        
        loss = loss_feat + self.lambda_f *loss_fgw
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

