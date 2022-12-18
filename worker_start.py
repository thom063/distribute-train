import torch
from torch import nn
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms

from distribute_train_struct.worker_node.worker_manager import WorkerManager
from lenet_model import lenet_model
from utils.grad_tools import load_grad,save_grad

epochs = 2
device = torch.device("cpu")
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
    cache_path = "./cache"

    # 先训练一个模型
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criteon = nn.CrossEntropyLoss()

    worker = WorkerManager(model, optimizer, cache_path)

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            logits = model(data.to(device))
            loss = criteon(logits, target.to(device))

            worker.model_updata(loss)

            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
