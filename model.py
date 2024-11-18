import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple,OrderedDict

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

MOBILENETV1_CONV_LAYERS = [
        Conv(kernel=(3, 3), stride=2, depth=32),
        DepthSepConv(kernel=(3, 3), stride=1, depth=64),
        DepthSepConv(kernel=(3, 3), stride=2, depth=128),
        DepthSepConv(kernel=(3, 3), stride=1, depth=128),
        DepthSepConv(kernel=(3, 3), stride=2, depth=256),
        DepthSepConv(kernel=(3, 3), stride=1, depth=256),
        DepthSepConv(kernel=(3, 3), stride=2, depth=512),
        DepthSepConv(kernel=(3, 3), stride=1, depth=512),
        DepthSepConv(kernel=(3, 3), stride=1, depth=512),
        DepthSepConv(kernel=(3, 3), stride=1, depth=512),
        DepthSepConv(kernel=(3, 3), stride=1, depth=512),
        DepthSepConv(kernel=(3, 3), stride=1, depth=512),
        DepthSepConv(kernel=(3, 3), stride=2, depth=1024),
        DepthSepConv(kernel=(3, 3), stride=1, depth=1024)
    ]

class Conv2d(nn.Module):
    def __init__(self, kernel : tuple[int, int], stride : int, in_depth : int, out_depth : int):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=kernel, stride=stride, padding=(kernel[0]//2,kernel[1]//2), bias=False)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(num_features=out_depth, momentum=MobileNetConfig.batch_norm_momentum, eps=MobileNetConfig.batch_norm_epsilon)
    def forward(self,input):
        res = self.relu(self.batch_norm(self.conv2d(input)))
        return res

class SeparableConv2d(nn.Module):
    def __init__(self, kernel : tuple[int, int], stride : int, in_depth : int, out_depth : int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_depth, out_channels=in_depth, kernel_size=kernel, stride=stride, padding=(kernel[0]//2,kernel[1]//2), groups=in_depth, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=in_depth, momentum=MobileNetConfig.batch_norm_momentum, eps=MobileNetConfig.batch_norm_epsilon)
        self.pointwise = nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=(1,1), bias=False)
        self.relu = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_depth, momentum=MobileNetConfig.batch_norm_momentum, eps=MobileNetConfig.batch_norm_epsilon)
    def forward(self,input):
        res = self.relu(self.batch_norm1(self.depthwise(input)))
        res = self.relu(self.batch_norm2(self.pointwise(res)))
        return res

class GlobalAvgPool(nn.Module):
    def __init__(self, kernel : tuple[int, int]):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel)
    def forward(self,input):
        res = self.avg_pool(input)
        return res

class Dropout(nn.Module):
    def __init__(self, keep_prob : float):
        super().__init__()
        self.drop = nn.Dropout(p=1-keep_prob)
    def forward(self,input):
        res = self.drop(input)
        return res

class Linear(nn.Module):
    def __init__(self, in_shape : int, out_shape : int):
        super().__init__()
        self.linear = nn.Linear(in_features=in_shape,out_features=out_shape)
    def forward(self,input):
        res = self.linear(input)
        return res

class MobileNetConfig:
    num_classes : int = 1001
    depth_multiplier : float = 1.0
    dropout_keep_prob : float = 0.999
    input_shape : tuple[int, int] = 224, 224
    channels : int = 3
    last_channels : int = 1024
    num_of_layers_stride2 : int = 5
    batch_norm_momentum : float = 0.9997
    batch_norm_epsilon: float = 0.001

class MobileNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        cur_channels = 3
        i = 1
        j = 1
        for layer in MOBILENETV1_CONV_LAYERS:
            if isinstance(layer, Conv):
                layers.append(("Conv2d_" + str(i) , Conv2d(layer.kernel, layer.stride, cur_channels, int(layer.depth * self.config.depth_multiplier))))
                i += 1
            elif isinstance(layer, DepthSepConv):
                layers.append(("SeparableConv2d_" + str(j), SeparableConv2d(layer.kernel, layer.stride, cur_channels, int(layer.depth * self.config.depth_multiplier))))
                j += 1
            else:
                raise "BullShitError"
            cur_channels = int(layer.depth * self.config.depth_multiplier)

        self.layers = nn.Sequential(OrderedDict(layers))
        self.avg_pool = GlobalAvgPool((self.config.input_shape[0]//(2**self.config.num_of_layers_stride2),self.config.input_shape[1]//(2**self.config.num_of_layers_stride2)))
        self.dropout = Dropout(keep_prob=self.config.dropout_keep_prob)
        self.linear = Linear(self.config.last_channels,self.config.num_classes)


    def forward(self, inputs, targets=None): # input.shape (B, C, H, W)  Batch_size, Channels, Height, Width
        res = self.layers(inputs)
        res = self.avg_pool(res)
        res = res.view(res.shape[0],res.shape[1])
        logits = self.linear(self.dropout(res))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

mob = MobileNet(MobileNetConfig)
param_count = 0
for name,param in mob.named_parameters():
    cur = 1
    for shape in param.shape:
        cur *= shape
    param_count += cur
    print(name, param.shape)
print(param_count)