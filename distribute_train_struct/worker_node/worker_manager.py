import abc
import glob
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
from utils.file_tools import save_buff, load_file, get_files


class WorkerManager(metaclass=abc.ABCMeta):
    worker_node: WorkerNode
    node_file_suffix: str = '.node'
    worker_file_suffix: str = '_worker'+node_file_suffix
    master_file_suffix: str = '_master'+node_file_suffix
    worker_cache: str
    master_cache: str
    model_name_record: list = []
    def __init__(self, model:nn.Module, optim: Optimizer, worker_cache:str, master_cache:str):
        self.model = WorkerNode(model, optim)
        self.optim = optim
        self.worker_cache = worker_cache
        self.master_cache = master_cache

        # 初始化模型数据
        self.updata_form_master()
        logging.info("worker初始化成功")

    # @abc.abstractmethod
    # def model_train(self):
    #     pass

    def broadcast_model(self):
        """
        广播模型，并更新
        """
        self.optim.zero_grad()
        params_list: List[TensorCache] = self.model.get_params()

        # 保存梯度
        grad_file_path: str = os.path.join(self.worker_cache, str(uuid.uuid1()) + self.worker_file_suffix)
        save_buff(params_list, grad_file_path)

        # 等待梯度被处理完毕
        while os.path.exists(grad_file_path):
            time.sleep(0.001)

        master_file = self.updata_form_master()
        self.model_name_record += master_file
        if len(self.model_name_record) > 20:
            # 删除前5缓存
            del self.model_name_record[:5]

    def updata_form_master(self):
        master_file = self.get_master_files()
        # 等待master文件出现
        while len(master_file) == 0:
            master_file = self.get_master_files()
            time.sleep(0.001)

        def updata_model(path:str):
            self.updata_param(path)
        list(map(updata_model, master_file))
        return master_file

    def updata_param(self, param_path):
        # 读取并更新最新参数
        while not os.path.exists(param_path):
            time.sleep(0.001) # 1ms
        params: List[TensorCache] = load_file(param_path)
        self.model.updata_parameter(params)

    def get_master_files(self):
        """
        获取master的文件
        :return: 文件列表
        """
        return list(filter(lambda file: file.endswith(self.master_file_suffix) and file not in self.model_name_record, get_files(self.master_cache)))

