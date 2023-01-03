# encoding: utf-8
# 主节点模型
from typing import List

import torch
from torch import nn
from torch.optim import Optimizer

from utils.tensor_cache import TensorCache



class MasterNode:
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

    # def updata(self, grads: List[TensorCache]):
    #     """
    #     根据梯度更新参数
    #     :return:
    #     """
    #     # 清空梯度
    #     self.optim.zero_grad()
    #     grad_maps = {cache.get_key(): torch.from_numpy(cache.get_tensor()) for cache in grads}
    #     for name, param in list(self.model.named_parameters()):
    #         param.grad = grad_maps.get(name)
    #     self.optim.step()

    def get_parameter(self):
        """
        获取模型参数
        :return: tensor列表形式返回
        """
        return [TensorCache(n, w.data) for n,w in list(self.model.named_parameters())]

    def updata_v2(self, params: List[TensorCache]):
        """
        取参数均值
        :return:
        """
        # 清空梯度
        self.optim.zero_grad()
        data_maps = {cache.get_key(): torch.from_numpy(cache.get_tensor()) for cache in params}
        for name, param in list(self.model.named_parameters()):
            param.data = data_maps.get(name)
        self.optim.step()

    def updata(self, params: List[TensorCache]):
        """
        取参数均值
        :return:
        """
        # 清空梯度
        self.optim.zero_grad()
        data_maps = {cache.get_key(): cache.get_tensor() for cache in params}
        for name, param in list(self.model.named_parameters()):
            param.data.copy_(data_maps.get(name))
        self.optim.step()