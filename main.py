import torch
from torch import nn
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms

from utils.grad_tools import load_grad,save_grad

epochs = 1
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


class simple_model(nn.Module):
    def __init__(self):
        super(simple_model, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 5))
        )
        self.classifer = nn.Sequential(
            nn.Linear(64 * 5 * 5, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x


def train(model, mode="train"):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    criteon = nn.CrossEntropyLoss()


    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if mode == "train":
                logits = model(data.to(device))
                loss = criteon(logits, target.to(device))
                optimizer.zero_grad()
                loss.backward()

                # # 保存测试
                # save_grad("./test", model)
                # optimizer.zero_grad()
                # a = list(model.named_parameters())[0][1]
                # d = torch.zeros_like(a.data)
                # a.data = d
                #
                # load_grad("./test", model)

                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                logits = model(data.to(device))
                test_loss += criteon(logits, target.to(device)).item()
                pred = logits.data.max(1)[1]
                target = target.cpu()
                pred = pred.cpu()
                correct += pred.eq(target.data).sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    if mode == "train":
        torch.save(model.state_dict(), "./model/origin_model.pth")


if __name__ == '__main__':
    model = simple_model()
    # 先训练一个模型
    train(model, mode="train")
    model.load_state_dict(torch.load("./model/origin_model.pth"))
    # 打印一下看看
    # summary(model, (3, 64, 64), device="cpu")
    # 输出一下精度
    print("----------------原模型精度-----------------")
    train(model, mode="val")
    # 定义剪枝配置
    config_list = [{'sparsity': 0.2, 'op_types': ['Conv2d']}]
    # 生成剪枝后的模型以及掩膜
    # 有很多种剪枝方法，可以自己选
    # pruner = L2FilterPruner(model, config_list)
    # model = pruner.compress()
    # pruner.export_model(model_path="./model/prune.pth", mask_path="./model/mask.pth")
    # # 压缩模型
    # pruner._unwrap_model()
    # m_Speedup = ModelSpeedup(model, torch.randn([64, 3, 64, 64], device=device), "./model/mask.pth", "cuda:0")
    # m_Speedup.speedup_model()

    import torch_pruning
    strategy = torch_pruning.strategy.L1Strategy()
    DG = torch_pruning.DependencyGraph()  # input_size是网络的输入大小
    # dummyInput = torch.randint(1, 30, (1,3))
    DG = DG.build_dependency(model, example_inputs=torch.randn([64, 3, 64, 64]))

    excluded_layers = list(model.modules())[-1:] #  list(self.model.modules())[:4] + list(self.model.modules())[-4:]
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m not in excluded_layers:
            pruning_plan = DG.get_pruning_plan(m, torch_pruning.prune_conv, idxs=strategy(m.weight, amount=0.8, round_to=8))
            pruning_plan.exec()



    # 打印一下模型
    # summary(model, (3, 64, 64), device="cpu")
    # 打印一下模型精度
    print("---------------剪枝模型精度------------------")
    train(model, mode="val")
    # 再次训练微调模型
    train(model, mode="train")
    # 打印一下精度
    print("---------------剪枝微调精度------------------")
    train(model, mode="val")
    # 保存模型
    torch.save(model.state_dict(), "./model/prune_model.pth")