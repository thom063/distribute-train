import torch


class TensorCache:
    _name: str
    _tensor: torch.Tensor
    def __init__(self, name, tensor):
        self._tensor = tensor
        self._name = name
    def get_key(self):
        return self._name
    def get_tensor(self):
        return self._tensor