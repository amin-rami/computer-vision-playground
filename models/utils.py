import json
from typing import Union, Tuple, List

from ._base import BaseModule
from .vggnet import VGGNet
from .resnet import ResNet


def from_config_file(filename: str):
    with open(filename) as file:
        config = json.load(file)
    return from_config(config)


def from_config(config: Union[dict, List[dict]]) -> Union[BaseModule, Tuple[BaseModule]]:
    if isinstance(config, dict):
        return from_dict(config)
    models = []
    for conf in config:
        models.append(from_dict(conf))
    return tuple(models)


def from_dict(d: dict) -> BaseModule:
    cls_map = {
        "ResNet": ResNet,
        "VGGNet": VGGNet,
    }
    cls = cls_map[d["class"]]
    params = d["parameters"]
    model = cls(**params)
    return model
