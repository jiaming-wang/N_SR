#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-04-26 17:01:43
LastEditTime: 2021-04-26 17:15:48
Description: fbatch_size=16, patch_size=48, L1 loss, epoch=300, ADAM, decay=150, lr=1e-4
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
# from torchvision.transforms import *
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return adder2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return adder2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class add_ResnetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, scale=0.1):
        super(add_ResnetBlock, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)
        out = self.scale * out + identity

        return out

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, scale_factor, args):
        super(Net, self).__init__()

        base_filter = 256
        n_resblocks = 32

        # self.sub_mean = MeanShift(args['data']['rgb_range'])
        # self.add_mean = MeanShift(args['data']['rgb_range'], sign=1)

        self.head = conv3x3(num_channels, base_filter)

        body = [
            add_ResnetBlock(base_filter, base_filter) for _ in range(n_resblocks)
        ]

        body.append(conv3x3(base_filter, base_filter))

        self.up = Upsampler(scale_factor, base_filter, activation=None)
        self.output_conv = conv3x3(base_filter, num_channels)
        self.body = nn.Sequential(*body)
        self.relu = nn.ReLU(inplace=True)

        # for m in self.modules():
        #     classname = m.__class__.__name__
        #     if classname.find('Conv2d') != -1:
        # 	    # torch.nn.init.kaiming_normal_(m.weight)
        # 	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # 	    if m.bias is not None:
        # 		    m.bias.data.zero_()
        #     elif classname.find('ConvTranspose2d') != -1:
        # 	    # torch.nn.init.kaiming_normal_(m.weight)
        # 	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # 	    if m.bias is not None:
        # 		    m.bias.data.zero_()

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.relu(self.head(x))
        res = x
        x = self.relu(self.body(x))
        x =  res + x

        x = self.up(x)
        x = self.output_conv(x)
        #x = self.add_mean(x)

        return x    

if __name__ == "__main__":
    input = Variable(torch.FloatTensor(1, 3, 40, 40))
    model = Net(num_channels= 3,base_filter=1,scale_factor= 1,args= 1)
    out = model(input)
    print(out.shape)