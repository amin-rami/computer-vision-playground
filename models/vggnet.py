import torch
import torch.nn as nn

from typing import List

from ._base import BaseModule


__all__ = [
    'VGGNet',
]


class VGGNet(BaseModule):
    def __init__(self, conv_layers: List[dict], fc_layers: List[dict], height: int, width: int, name: str = None):
        super().__init__(name)
        self._conv_layer_conf = conv_layers
        self._fc_layer_conf = fc_layers
        self._height = height
        self._width = width
        self.conv_layers = self._make_conv_layers()
        self.flatten = nn.Flatten()
        self.fc_layers = self._make_fc_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

    def _make_conv_layers(self) -> nn.Sequential:
        conv_layers = []
        for layer_conf in self.conv_layer_conf:
            in_ch = layer_conf['in_channels']
            out_ch = layer_conf['out_channels']
            batch_normalization = layer_conf.get("batch_normalization")
            max_pooling = layer_conf.get("max_pooling")
            p = layer_conf.get("dropout")
            num = layer_conf.get("num")
            num = num if num else 1

            layers = []
            for i in range(num):
                if i != 0:
                    in_ch = out_ch
                layers.append(nn.Conv2d(in_ch, out_ch, 3, padding='same'))
                if batch_normalization:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU())
                if max_pooling and i == num - 1:
                    layers.append(nn.MaxPool2d(2, 2))
                if p:
                    layers.append(nn.Dropout2d(p))
            conv_layers.append(nn.Sequential(*layers))
        return nn.Sequential(*conv_layers)

    def _make_fc_layers(self) -> nn.Sequential:
        fc_layers = []
        num = len(self.fc_layer_conf)
        features_in = self._get_conv_out_features()

        for i, layer_conf in enumerate(self.fc_layer_conf):
            features_out = layer_conf["features_out"]
            p = layer_conf.get("dropout")

            fc_layers.append(nn.Linear(features_in, features_out))
            if i + 1 != num:
                fc_layers.append(nn.ReLU())
            if p:
                fc_layers.append(nn.Dropout(p))
            features_in = features_out
        return nn.Sequential(*fc_layers)

    def _get_conv_out_features(self) -> int:
        downsamples = 0
        for layer in self.conv_layer_conf:
            if not layer.get("max_pooling"):
                continue
            downsamples += 1
        h = self.height // (2 ** downsamples)
        w = self.width // (2 ** downsamples)
        k = self.conv_layer_conf[-1]["out_channels"]
        return h * w * k

    @property
    def conv_layer_conf(self):
        return self._conv_layer_conf

    @property
    def fc_layer_conf(self):
        return self._fc_layer_conf

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width
