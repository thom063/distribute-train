import argparse
import logging

import torch
from torch import nn
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms

from distribute_train_struct.worker_node.worker_manager import WorkerManager
from lenet_model import lenet_model
from utils.grad_tools import load_grad,save_grad

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s -%(message)s')

epochs = 2
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor()
                     ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False,
                     transform=transforms.Compose([
                         transforms.ToTensor()
                     ])),
    batch_size=batch_size, shuffle=True)



if __name__ == '__main__':
    model = lenet_model()

    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_cache_path', default='./cache/1', type=str, required=False)
    parser.add_argument('--master_cache_path', default='./cache', type=str, required=False)
    parser.add_argument('--device', default='cpu', type=str, required=False)
    args = parser.parse_args()
    worker_cache_path = args.worker_cache_path
    master_cache_path = args.master_cache_path

    device = torch.device(args.device)
    # 先训练一个模型
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criteon = nn.CrossEntropyLoss()

    worker = WorkerManager(model, optimizer, worker_cache_path, master_cache_path)

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            logits = model(data.to(device))
            loss = criteon(logits, target.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                worker.broadcast_model()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))