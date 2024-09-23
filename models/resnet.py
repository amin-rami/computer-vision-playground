import torch.nn as nn


__all__ = [
    'ResNet',
]


class ResNet(nn.Module):
    def __init__(self, conv_layers, fc_layers, height, width):
        super().__init__()
        self._conv_layer_conf = conv_layers
        self._fc_layer_conf = conv_layers
        self._height = height
        self._wdith = width
        self.conv_layers = self._make_conv_layers()
        self.flatten = nn.Flatten()
        self.fc_layers = self._make_fc_layers()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

    def _make_conv_layers(self):
        conv_layers = []
        for layer in self.conv_layer_conf:
            layers = []
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]
            b_norm = layer.get("b_norm")
            max_pool = layer.get("max_pool")
            num = layer.get("num")
            num = num if num else 1

            for i in range(num):
                if i != 0:
                    in_channels = out_channels
                layers.append(_BasicBlock(
                    in_channels,
                    out_channels,
                    b_norm,
                    max_pool and i + 1 == num
                ))
            conv_layers.append(nn.Sequential(*layers))
        return nn.Sequential(*conv_layers)

    def _make_fc_layers(self):
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

    def _get_conv_out_features(self):
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
    def __init__(self, in_channels, out_channels, b_norm=None, max_pooling=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(out_channels) if b_norm else b_norm
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
        self.batch_norm2 = nn.BatchNorm2d(out_channels) if b_norm else b_norm
        self.max_pooling = nn.MaxPool2d(2, 2) if max_pooling else max_pooling
        self.relu = nn.ReLU()
        if in_channels != self.out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, padding='same')

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels:
            residual = self.projection(residual)

        x = self.conv2(x)
        if self.batch_norm1:
            x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)

        x = x + residual
        x = self.relu(x)
        if self.max_pooling:
            x = self.max_pooling(x)
        return x
