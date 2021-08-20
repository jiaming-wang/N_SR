#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-06-02 21:02:40
LastEditTime: 2021-08-20 23:54:25
@Description: batch_size=16, patch_size=48, L1 loss, epoch=1000, lr=1e-4, decay=200, ADAM
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
# from torchvision.transforms import *

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        self.args = args
        num_channels = self.args['data']['batch_size']
        scale_factor = self.args['data']['upsacle']
        
        self.sub_mean = MeanShift(args['data']['rgb_range'])
        self.add_mean = MeanShift(args['data']['rgb_range'], sign=1)
        base_filter = 64
        n_resgroups = 10
        n_resblocks = 20
        reduction = 16 #number of feature maps reduction
        self.head = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='relu', norm=None)

        body = [
            ResidualGroup(base_filter, 3, reduction, act=nn.ReLU(True), res_scale=scale_factor, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        
        body.append(ConvBlock(base_filter, base_filter, 3, 1, 1, activation='relu', norm=None))

        self.up = Upsampler(scale_factor, base_filter, activation=None)
        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation='relu', norm=None)

        self.body = nn.Sequential(*body)
    
    def forward(self, x):
        
        x = self.head(x)
        
        res = self.body(x)
        res = res + x
        
        x = self.up(res)
        x = self.output_conv(x)

        return x

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(ConvBlock(n_feat, n_feat, 3, 1, 1, activation=None, norm=None))
            # if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res = res + x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                n_feat, kernel_size, reduction, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(ConvBlock(n_feat, n_feat, 3, 1, 1, activation=None, norm=None))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res
