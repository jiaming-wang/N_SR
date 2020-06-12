#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-06-05 11:36:25
@LastEditTime: 2020-06-07 09:23:40
@Description: path_size = 96, batch_size = 32, epoch = 1000, L1, the RCAB block is the same as RCAN.
'''
import os
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch
import numpy as np

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, scale_factor, args):
        super(Net, self).__init__()

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = MeanShift(args['data']['rgb_range'])
        # self.add_mean = MeanShift(args['data']['rgb_range'], sign=1)

        self.upsample = nn.Upsample(scale_factor=scale_factor,
                                    mode='bicubic', align_corners=False)
        # DRN-s
        # if scale == 4, n_res = 30, base_filter = 16
        # if scale == 8, n_res = 30, base_filter = 8
        # DRN-l
        # if scale == 4, n_res = 40, base_filter = 20
        # if scale == 8, n_res = 36, base_filter = 10
        n_res = 40
        base_filter = 20
        phase = 1
        kernel_size = 3
        reduction = 16 #number of feature maps reduction

        self.head = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='relu', norm=None)
        
        self.down = [
            DownBlock_drn(2, base_filter * pow(2, p), base_filter * pow(2, p), base_filter * pow(2, p + 1)) for p in range(phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = []
        # remove data augmentation block
        # up_body_blocks = [[
        #     RCAB(
        #         n_feat, kernel_size, reduction, act=nn.ReLU(True), res_scale=1
        #         ) for _ in range(n_res)
        # ]for p in range(self.phase, 1, -1)
        # ]

        up_body_blocks.insert(0, [
            RCAB(base_filter, kernel_size, reduction, act=nn.ReLU(True), res_scale=1) for _ in range(n_res)
        ])

        up = [[
            Upsampler(2, base_filter * pow(2, phase), False, False, False),
            ConvBlock(base_filter * pow(2, phase), base_filter * pow(2, phase-1), kernel_size=1, stride=0 ,padding=0, activation=None)
        ]]

        # remove data augmentation block
        # The rest upsample blocks
        # for p in range(phase - 1, 0, -1):
        #     up.append([
        #         Upsampler(2, base_filter * pow(2, phase), False, False, False),
        #         ConvBlock(base_filter * pow(2, phase), base_filter * pow(2, phase-1), kernel_size=1, stride=0 ,padding=0, activation=None)
        #     ])

        self.up_blocks = nn.ModuleList()
        for idx in range(phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )
        
        # tail conv that output sr imgs
        tail = [ConvBlock(base_filter * pow(2, phase), 3, kernel_size=3, activation=None)]

        # for p in range(phase, 0, -1):
        #     tail.append(
        #         ConvBlock(base_filter * pow(2, phase), 3, kernel_size=3, activation=None)
        #     )
        self.tail = nn.ModuleList(tail)

    def forward(self, x):
        phase = 1
        # upsample x to target sr size
        x = self.upsample(x)

        # preprocess
        # x = self.sub_mean(x)
        x = self.head(x)

        # down phases,
        copies = []
        for idx in range(phase):
            copies.append(x)
            x = self.down[idx](x)

        # up phases
        sr = self.tail[0](x)
        sr = self.add_mean(sr)
        results = [sr]
        for idx in range(phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[phase - idx - 1]), 1)
            # output sr imgs
            sr = self.tail[idx + 1](x)
            # sr = self.add_mean(sr)

            results.append(sr)
        print(len(results))
        return results


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
        
class DownBlock_drn(nn.Module):
    def __init__(self, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock_drn, self).__init__()
        negval = 0.2
     
        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x