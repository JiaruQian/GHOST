import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="cora")
current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
parser.add_argument("--dataset_dir", type=str, default=root_dir)

log_path = os.path.join(os.path.dirname(current_path), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)

parser.add_argument("--logs_dir", type=str, default=log_path)
parser.add_argument("--specified_domain_skew_task", type=str, default=None)
parser.add_argument("--task", type=str, default="node_classification")
parser.add_argument("--skew_type", type=str, default="label_skew")
parser.add_argument("--train_val_test_split", type=list, default=[0.2, 0.4, 0.4])
parser.add_argument("--dataset_split_metric", type=str, default="transductive")

parser.add_argument("--num_rounds", type=int, default=1)
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--T_L", type=int, default=100)
parser.add_argument("--cl_sample_rate", type=float, default=1.0)
parser.add_argument("--evaluation_mode", type=str, default="global")
parser.add_argument("--fed_algorithm", type=str, default="GHOST")
parser.add_argument("--model", type=str, default="GCN")
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--f_scale", type=float, default=1)
parser.add_argument("--r_scale", type=float, default=1e-11)
parser.add_argument("--n_scale", type=float, default=1e-11)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=4e-4)

parser.add_argument("--noise_dim", type=int, default=64)
parser.add_argument("--M", type=int, default=3)
parser.add_argument("--T_G", type=int, default=5)
parser.add_argument("--lambda_d", type=float, default=0.05)
parser.add_argument("--lambda_f", type=float, default=0.1)
parser.add_argument("--lambda_r", type=float, default=1)
parser.add_argument("--lambda_n", type=float, default=1)


parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--dirichlet_alpha", type=float, default=0.05)
parser.add_argument("--least_samples", type=int, default=5)
parser.add_argument("--dirichlet_try_cnt", type=int, default=10000)





args = parser.parse_args()
