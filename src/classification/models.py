import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Linear(nn.Module):
    def __init__(self, in_channels: int = 1024, num_classes: int = 2):
        super().__init__()
        flatten = True
        if flatten:
            self.FlatFeats = nn.Flatten()
            self.fc = nn.Linear(in_channels, num_classes)
        else:
            self.fc = nn.Linear(in_channels, num_classes)


    def forward(self, xb):
        flatten = True
        if flatten:
            xb = self.FlatFeats(xb)
        # xb = xb.view(xb.size(0), -1)
        return self.fc(xb)


# adjusted from https://github.com/mrchntia/ScaleNorm/blob/main/image_classification/resnet9.py
def conv_block(in_channels: int, out_channels: int, pool: bool = False, act_func: nn.Module = nn.Mish, num_groups: int = 32):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(min(num_groups, out_channels), out_channels),
        act_func()
        ]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        act_func: nn.Module = nn.Mish,
        scale_norm: bool = False,
        num_groups: tuple[int, ...] = (32, 32, 32, 32),
        ):
        super().__init__()
        if num_classes == 2:
            num_classes = 1

        self.num_classes = num_classes

        assert (
            isinstance(num_groups, tuple) and len(num_groups) == 4
        ), 'num_groups must be a tuple with 4 members'
        groups = num_groups

        self.conv1 = conv_block(
            in_channels, 64, act_func=act_func, num_groups=groups[0]
        )
        self.conv2 = conv_block(
            64, 128, pool=True, act_func=act_func, num_groups=groups[0]
        )

        self.res1 = nn.Sequential(
            *[
                conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
                conv_block(128, 128, act_func=act_func, num_groups=groups[1]),
            ]
        )

        self.conv3 = conv_block(
            128, 256, pool=True, act_func=act_func, num_groups=groups[2]
        )
        self.conv4 = conv_block(
            256, 256, pool=True, act_func=act_func, num_groups=groups[2]
        )

        self.res2 = nn.Sequential(
            *[
                conv_block(256, 256, act_func=act_func, num_groups=groups[3]),
                conv_block(256, 256, act_func=act_func, num_groups=groups[3]),
            ]
        )

        self.MP = nn.AdaptiveMaxPool2d((2, 2))
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(1024, num_classes)
        self.sigmoid = nn.Sigmoid()

        if scale_norm:
            self.scale_norm_1 = nn.GroupNorm(min(num_groups[1], 128), 128)
            self.scale_norm_2 = nn.GroupNorm(min(groups[3], 256), 256)
        else:
            self.scale_norm_1 = nn.Identity()
            self.scale_norm_2 = nn.Identity()


    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.scale_norm_1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.scale_norm_2(out)
        out = self.MP(out)
        out_emb = self.FlatFeats(out)
        out = self.classifier(out_emb)
        if self.num_classes == 1:
            out = self.sigmoid(out)
        return out