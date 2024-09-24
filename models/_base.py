import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name if name else f"{type(self).__name__}{id(self)}"
