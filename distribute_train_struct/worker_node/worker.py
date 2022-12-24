from typing import List

import torch
from torch import nn
from torch.optim import Optimizer

from utils.tensor_cache import TensorCache


class WorkerNode:
    # 主节点模型，复制存储与更新参数并分发参数
    model: nn.Module
    # 优化器
    optim: Optimizer
    def __init__(self, model: nn.Module, optim: Optimizer):
        """
        init
        :param model: 模型
        :param optim: 优化器
        """
        self.model = model
        self.optim = optim
        self.device = self.get_device(self.model)

    def updata_parameter(self, params: List[TensorCache]):
        """
        根据master节点提供的参数更新模型参数
        :return:
        """
        grad_maps = {cache.get_key(): cache.get_tensor() for cache in params}
        with torch.no_grad():
            for name, param in list(self.model.named_parameters()):
                param.copy_(grad_maps.get(name))

    def get_grad(self, loss: torch.Tensor) -> List[TensorCache]:
        """
        获取模型参数
        :return: tensor列表形式返回
        """
        # 清空梯度
        self.optim.zero_grad()
        loss.backward()
        gradTensors = [TensorCache(n, w.grad.numpy()) for w, n in zip(self.model.parameters(), self.model.state_dict())]
        return gradTensors

    # fixme 后期换成static 不用param
    def get_params(self) -> List[TensorCache]:
        """
        传播变量
        """
        return [TensorCache(n, w.data.cpu().numpy()) for w, n in zip(self.model.parameters(), self.model.state_dict())]


    def get_device(self, model):
        return list(model.parameters())[0].device