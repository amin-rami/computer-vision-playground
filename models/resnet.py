import torch
import torch.nn as nn

from typing import List

from ._base import BaseModule


__all__ = [
    'ResNet',
]


class ResNet(BaseModule):
    def __init__(self, conv_layers: List[dict], fc_layers: List[dict], height: int, width: int, name: str = None):
        super().__init__(name)
        self._conv_layer_conf = conv_layers
        self._fc_layer_conf = fc_layers
        self._height = height
        self._wdith = width
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
            layers = []
            in_channels = layer_conf["in_channels"]
            out_channels = layer_conf["out_channels"]
            b_norm = layer_conf.get("batch_normalization")
            max_pool = layer_conf.get("max_pooling")
            dropout = layer_conf.get("dropout")
            num = layer_conf.get("num")
            num = num if num else 1

            for i in range(num):
                if i != 0:
                    in_channels = out_channels
                layers.append(_BasicBlock(
                    in_channels,
                    out_channels,
                    b_norm,
                    max_pool and i + 1 == num,
                    dropout
                ))
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
                fc_layers.append(nn.Dropout())
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
        return self._wdith


class _BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, b_norm: bool = None, max_pooling: bool = None, p: float = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(out_channels) if b_norm else b_norm
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
        self.batch_norm2 = nn.BatchNorm2d(out_channels) if b_norm else b_norm
        self.max_pooling = nn.MaxPool2d(2, 2) if max_pooling else max_pooling
        self.dropout = nn.Dropout2d(p) if p else p
        self.relu = nn.ReLU()
        if in_channels != self.out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, padding='same')

    def forward(self, x: torch.Tensor):
        residual = x
        if self.in_channels != self.out_channels:
            residual = self.projection(residual)

        x = self.conv1(x)
        if self.dropout:
            x = self.dropout(x)
        if self.batch_norm1:
            x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.dropout:
            x = self.dropout(x)
        if self.batch_norm2:
            x = self.batch_norm2(x)

        x = x + residual
        x = self.relu(x)
        if self.max_pooling:
            x = self.max_pooling(x)
        return x
