import random

import numpy as np
import torch.nn.functional as F
import torch


class BaseServer:
    def __init__(self, args, clients, model, data, logger):
        self.logger = logger
        self.sampled_clients = None
        self.clients = clients
        self.model = model
        self.cl_sample_rate = args.cl_sample_rate
        self.num_rounds = args.num_rounds
        self.T_L = args.T_L
        self.data = data
        self.num_total_samples = sum([client.num_samples for client in self.clients])

    def run(self):
        for round in range(self.num_rounds):
            print("round "+str(round+1)+":")
            self.logger.write_round(round+1)
            self.sample()
            self.communicate()

            print("cid : ", end='')
            for cid in self.sampled_clients:
                print('---------------------------')
                print(f'{cid} training proxy model...')
                for epoch in range(self.T_L):
                    self.clients[cid].train()
    
            self.aggregate()
            #generalized setting
            self.global_evaluate()

    def communicate(self):
        for cid in self.sampled_clients:
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)

    def sample(self):
        num_sample_clients = int(len(self.clients) * self.cl_sample_rate)
        sampled_clients = random.sample(range(len(self.clients)), num_sample_clients)
        self.sampled_clients = sampled_clients

    def aggregate(self):
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        for i, cid in enumerate(self.sampled_clients):
            w = self.clients[cid].num_samples / num_total_samples
            for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                if i == 0:
                    global_param.data.copy_(w * client_param)
                else:
                    global_param.data += w * client_param

    def global_evaluate(self):
        self.model.eval()
        with torch.no_grad():
            _,out = self.model(self.data)
            loss = F.nll_loss(out[self.data.test_mask], self.data.y[self.data.test_mask])
            pred = out[self.data.test_mask].max(dim=1)[1]
            acc = pred.eq(self.data.y[self.data.test_mask]).sum().item() / self.data.test_mask.sum().item()
            print("test_loss : "+format(loss.item(), '.4f'))
            self.logger.write_test_loss(loss.item())
            print("test_acc : "+format(acc, '.4f'))
            self.logger.write_test_acc(acc)


class BaseClient:
    def __init__(self, args, model, data):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.data = data
        self.loss_fn = F.nll_loss
        self.num_samples = len(data.x)
        self.args = args

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        _, out = self.model(self.data)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
