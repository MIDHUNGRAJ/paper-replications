# ResNet-34 model (He et al., 2015). ImageNet stem: 224x224 input.

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Type


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False)


class BasicBlock(nn.Module):
    """Two 3x3 conv layers with a skip connection."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        batchnorm: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if batchnorm is None:
            batchnorm = nn.BatchNorm2d

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = batchnorm(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = batchnorm(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # residual connection
        return self.relu(out)


class ResNet(nn.Module):
    """ResNet backbone. Pass BasicBlock and layer counts, e.g. [3, 4, 6, 3] for ResNet-34."""

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        batchnorm: Optional[Callable[..., nn.Module]] = None,
        num_classes: int = 1000,
    ):
        super().__init__()
        if batchnorm is None:
            batchnorm = nn.BatchNorm2d
        self._batchnorm = batchnorm

        self._in_channels = 64

        # ImageNet-style stem (7x7 conv + max pool). bias=False because BatchNorm handles the bias term.
        self.conv1 = nn.Conv2d(3, self._in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = batchnorm(self._in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], channels: int, blocks: int, stride: int = 1):
        batchnorm = self._batchnorm
        downsample = None

        if stride != 1 or self._in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self._in_channels, channels * block.expansion, stride),
                batchnorm(channels * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(block(self._in_channels, channels, stride, downsample, batchnorm))
        self._in_channels = channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self._in_channels, channels, batchnorm=batchnorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
