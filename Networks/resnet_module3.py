import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


"""
MODEL
"""


def conv3x3(in_channels, out_channels, stride=1, groups=1, bias=False):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=bias)


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=bias)


def conv5x5(in_channels, out_channels, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, groups=groups, bias=bias)


def conv7x7(in_channels, out_channels, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, groups=groups, bias=bias)


class SEModule(nn.Module):    # Squeeze-and-Excitation Network(SE-Net) Channel Attention
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     # global average pooling
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)   # Fully Concatenation 1
        self.relu = nn.ReLU(inplace=True)                                                # ReLU
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)   # Fully Concatenation 2
        self.sigmoid = nn.Sigmoid()                                                       # sigmoid

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x
        # return x


class SpatialAttModule(nn.Module):  #  Spatial Attention
    def __init__(self, channels, reduction=16):
        super(SpatialAttModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=3, padding=1)  # Convolution 1
        self.relu = nn.ReLU(inplace=True)  # ReLU
        self.conv2 = nn.Conv2d(channels // reduction, 1, kernel_size=3, padding=1)  # Convolution 2
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return input * x
        # return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scales=2, groups=1, norm_layer=None, se=True, sa=True, downsample=None):
        super(ResBlock, self).__init__()

        if out_channels % scales != 0:
            raise ValueError('Planes must be divisible by scales')

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d                        # BN layer
        bottleneck_planes = groups * out_channels
        channels_scale = bottleneck_planes // scales
        self.stride = stride
        self.scales = scales
        self.downsample = downsample
        self.conv1 = conv3x3(in_channels, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.relu = nn.ReLU(inplace=True)    # inplace: Overwrite original value
        # self.modlist0 = nn.ModuleList([conv1x1(bottleneck_planes, bottleneck_planes, stride=stride),
        #                                norm_layer(bottleneck_planes)])
        self.modlist1 = nn.ModuleList([conv1x1(bottleneck_planes, channels_scale, stride=stride),
                                       norm_layer(channels_scale),
                                       nn.ReLU(inplace=True),
                                       conv3x3(channels_scale, channels_scale, stride=stride, groups=groups),
                                       norm_layer(channels_scale)])
        self.modlist2 = nn.ModuleList([conv1x1(bottleneck_planes, channels_scale, stride=stride),
                                       norm_layer(channels_scale),
                                       nn.ReLU(inplace=True),
                                       conv3x3(channels_scale, channels_scale, stride=stride, groups=groups),
                                       norm_layer(channels_scale),
                                       nn.ReLU(inplace=True),
                                       conv3x3(channels_scale, channels_scale, stride=stride, groups=groups),
                                       norm_layer(channels_scale)])

        self.se = SEModule(bottleneck_planes) if se else None
        self.sa = SpatialAttModule(bottleneck_planes) if sa else None

        self.conv2 = conv1x1(bottleneck_planes*2, bottleneck_planes, stride=stride)
        # self.bn2 = norm_layer(bottleneck_planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        in_branch0 = out
        in_branch1 = out
        # in_branch2 = out

        ys = []
        for s in range(self.scales):
            if s == 0:
                for m in self.modlist1:
                    in_branch0 = m(in_branch0)
                ys.append(in_branch0)
            elif s == 1:
                for m in self.modlist2:
                    in_branch1 = m(in_branch1)
                ys.append(in_branch1)
            # else:
            #     for m in self.modlist3:
            #         in_branch2 = m(in_branch2)
            #     ys.append(in_branch2)
        out = torch.cat(ys, 1)
        # out = in_branch0 + in_branch1 + in_branch2
        # out = self.conv2(out)
        # out = self.bn2(out)

        if self.se is not None:
            out_se = self.se(out)

        if self.sa is not None:
            out_sa = self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # out = out * out_se * out_sa
        out = torch.cat((out_se, out_sa), 1)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)

        return out


class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        img_channel = 3
        out_channels = 32
        out_channels2 = out_channels
        self.layer1 = conv3x3(img_channel, out_channels)
        self.layer2 = ResBlock(out_channels, out_channels, stride=1, groups=1, norm_layer=None, se=True,
                               sa=True, downsample=None)
        self.layer3 = ResBlock(out_channels, out_channels, stride=1, groups=1, norm_layer=None, se=True,
                               sa=True, downsample=None)
        self.layer4 = ResBlock(out_channels, out_channels, stride=1, groups=1, norm_layer=None, se=True,
                               sa=True, downsample=None)
        self.layer5 = ResBlock(out_channels2, out_channels2, stride=1, groups=1, norm_layer=None, se=True,
                               sa=True, downsample=None)
        self.layer6 = ResBlock(out_channels2, out_channels2, stride=1, groups=1, norm_layer=None, se=True,
                               sa=True, downsample=None)
        self.layer7 = ResBlock(out_channels2, out_channels2, stride=1, groups=1, norm_layer=None, se=True,
                               sa=True, downsample=None)
        self.layer8 = conv3x3(out_channels2, img_channel, stride=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output

    def tensor_max(self, tensor1, tensor2):
        max_tensor = torch.max(tensor1, tensor2)
        return max_tensor

    def tensor_cat(self, tensor1, tensor2):
        cat_tensor = torch.cat((tensor1, tensor2), 1)
        return cat_tensor

    def forward(self, input1, input2):
        tensor1 = self.forward_once(input1)
        tensor2 = self.forward_once(input2)
        output = self.tensor_max(tensor1, tensor2)
        output = self.layer5(output)
        output = self.layer6(output)
        output = self.layer7(output)
        output = self.layer8(output)
        output = self.sigmoid(output)
        return output


def MyRegressionNet():
    model = RegressionNet()
    return model