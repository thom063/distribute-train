import logging
import os
import shutil
import time
from typing import List

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
    master_file_name: str = 'master'+node_file_suffix
    cache_path: str
    def __init__(self, model:nn.Module, optim: Optimizer, cache_path:str):

        # 初始化缓存路径
        set_dir(cache_path)
        self.cache_path = cache_path
        # 初始化主节点
        self.master_node = MasterNode(model, optim)
        self.master_cache_dir = os.path.join(self.cache_path,self.master_file_name)

        # 保存参数
        cs.save_buff(self.master_node.get_parameter(), self.master_cache_dir)
        logging.info("init master")

        # 初始化监控器
    def monitor(self):
        # 监控是否存在梯度文件
        logging.info("start monitor")
        while 1:
            time.sleep(0.002) # 2ms
            def cache_file_deal(grad_file):
                grad_path = os.path.join(self.cache_path, grad_file)
                if grad_file.endswith(self.worker_file_suffix):
                    # 根据梯度文件更新模型
                    grad_list: List[TensorCache] = cs.load_file(grad_path)
                    self.master_node.updata_parameter(grad_list)
                    # 保存参数
                    cs.save_buff(self.master_node.get_parameter(), self.master_cache_dir)
                    # 清理缓存文件
                    os.remove(grad_path)

            list(map(cache_file_deal, os.listdir(self.cache_path)))
