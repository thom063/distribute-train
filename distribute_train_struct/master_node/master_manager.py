import logging
import os
import shutil
import time
import uuid
from typing import List

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
import utils.file_tools as cs
from distribute_train_struct.master_node.master import MasterNode
from utils.tensor_cache import TensorCache
from utils.file_tools import set_dir


class MasterManager:
    master_node: MasterNode
    node_file_suffix: str = '.node'
    worker_file_suffix: str = '_worker'+node_file_suffix
    master_file_suffix: str = '_master'+node_file_suffix
    worker_cache: str
    master_cache: str
    def __init__(self, model:nn.Module, optim: Optimizer, worker_cache:str, master_cache:str):

        # 初始化缓存路径
        list(map(set_dir, [worker_cache, master_cache]))

        self.worker_cache = worker_cache
        self.master_cache = master_cache
        # 初始化主节点
        self.master_node = MasterNode(model, optim)

        # 保存模型参数
        self.save_model_param()

        logging.info("init master")


    def monitor2(self, worker_num: int):
        """
        参数监控
        """
        # 监控是否存在梯度文件
        logging.info("start monitor")

        def load_cache_file(param_file: str):
            param_list: List[TensorCache] = cs.load_file(param_file)
            return param_list

        while 1:
            time.sleep(0.002) # 2ms

            # 读取参数
            worker_file = list(filter(lambda file:file.endswith(self.worker_file_suffix), cs.get_files(self.worker_cache)))
            if worker_num <= len(worker_file):
                logging.info("star updata model")
                param_list = list(map(load_cache_file, worker_file))
                param_dict = {}
                for param_tensors in param_list:
                    for param_tensor in param_tensors:
                        if param_tensor.get_key() not in param_dict.keys():
                            param_dict[param_tensor.get_key()] = []
                        param_dict[param_tensor.get_key()].append(param_tensor.get_tensor())
                param_update_list = []
                for param_name in param_dict.keys():
                    param_update_list.append(TensorCache(param_name, np.mean(np.concatenate([np.expand_dims(l,0) for l in param_dict.get(param_name)],0), 0)))
                self.master_node.updata_v2(param_update_list)

                # 清理之前的模型参数
                list(map(lambda f:os.remove(f), [f for f in cs.get_files(self.master_cache) if f.endswith(self.master_file_suffix)]))

                # 输出参数文件
                list(map(lambda f:os.remove(f), worker_file))

                # 保存最新模型
                self.save_model_param()
                logging.info("end updata model")

    def save_model_param(self):
        master_cache_dir = os.path.join(self.master_cache, str(uuid.uuid1()) + self.master_file_suffix)
        # 保存参数
        cs.save_buff(self.master_node.get_parameter(), master_cache_dir)

