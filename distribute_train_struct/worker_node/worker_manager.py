import abc
import logging
import os.path
import time
import uuid
from typing import List

import torch
from torch import nn
from torch.optim import Optimizer

from distribute_train_struct.worker_node.worker import WorkerNode
from utils.tensor_cache import TensorCache
from utils.file_tools import save_buff, load_file


class WorkerManager(metaclass=abc.ABCMeta):
    worker_node: WorkerNode
    node_file_suffix: str = '.node'
    worker_file_suffix: str = '_worker'+node_file_suffix
    master_file_name: str = 'master'+node_file_suffix
    cache_path: str
    def __init__(self, model:nn.Module, optim: Optimizer, cache_path:str):
        self.model = WorkerNode(model, optim)
        self.optim = optim
        self.cache_path = cache_path
        self.master_file_path = os.path.join(self.cache_path, self.master_file_name)

        # 初始化模型数据
        self.updata_param(self.master_file_path)
        logging.info("worker初始化成功")
    # @abc.abstractmethod
    # def model_train(self):
    #     pass

    def model_updata(self, loss: torch.Tensor):
        # 传递梯度
        grads: List[TensorCache] = self.model.get_grad(loss)
        # 保存梯度
        grad_file_path: str = os.path.join(self.cache_path, str(uuid.uuid1()) + self.worker_file_suffix)
        save_buff(grads, grad_file_path)

        # 等待梯度被处理完毕
        while os.path.exists(grad_file_path):
            time.sleep(0.001) # 1ms
        self.updata_param(self.master_file_path)

    def updata_param(self, master_file_path):
        # 读取并更新最新参数
        while not os.path.exists(master_file_path):
            time.sleep(0.001) # 1ms
        params: List[TensorCache] = load_file(master_file_path)
        self.model.updata_parameter(params)