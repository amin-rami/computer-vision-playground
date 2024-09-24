import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name if name else f"{type(self).__name__}{id(self)}"

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return x
