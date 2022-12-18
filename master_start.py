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

device = torch.device("cpu")
cache_path = "./cache"
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s -%(message)s')

if __name__ == '__main__':
    model = lenet_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    m = MasterManager(model, optimizer, cache_path)
    m.monitor()