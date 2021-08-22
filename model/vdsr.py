#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-18 15:19:38
LastEditTime: 2021-08-22 10:48:21
@Description: file content
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args
        num_channels = self.args['data']['n_colors']
        scale_factor = self.args['data']['upsacle']
        
        base_filter = 64
        self.head = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='relu', norm=None, bias = True)

        body = [
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='relu', norm=None, bias = True) for _ in range(18)
        ]

        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation='relu', norm=None, bias = True)
        self.body = nn.Sequential(*body)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    # torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    # torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def forward(self, x):
        
        res = x
        x = self.head(x)
        x = self.body(x)
        x = self.output_conv(x)
        x = torch.add(x, res)
        # print(x)
        # print(res)
        return x    