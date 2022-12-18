from typing import List
from torch import nn

import utils.file_tools as cs
from utils.tensor_cache import TensorCache


def save_grad(file_path: str, model: nn.Module):
    gradTensors = [TensorCache(n, w.grad) for w, n in zip(model.parameters(), model.state_dict())]
    cs.save_buff(gradTensors, file_path)

def load_grad(file_path: str, model: nn.Module):
    grad_list:List[TensorCache] = cs.load_file(file_path)
    grad_maps = {cache.get_key():cache.get_tensor() for cache in grad_list}
    for name, param in list(model.named_parameters()):
        param.grad = grad_maps.get(name)
