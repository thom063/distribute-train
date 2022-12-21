import argparse
import logging

import torch
from torch import nn
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms

from distribute_train_struct.master_node.master import MasterNode
from distribute_train_struct.master_node.master_manager import MasterManager
from lenet_model import lenet_model
from utils.grad_tools import load_grad,save_grad

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s -%(message)s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_cache_path', default='./cache', type=str, required=False)
    parser.add_argument('--master_cache_path', default='./cache', type=str, required=False)
    parser.add_argument('--device', default='cpu', type=str, required=False)
    args = parser.parse_args()
    worker_cache_path = args.worker_cache_path
    master_cache_path = args.master_cache_path
    device = torch.device(args.device)

    model = lenet_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    m = MasterManager(model, optimizer, worker_cache_path, master_cache_path)
    m.monitor2(2)