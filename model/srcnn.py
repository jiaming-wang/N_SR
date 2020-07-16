#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-06-16 15:19:38
@LastEditTime: 2020-07-16 16:38:16
@Description: batch-size = 64, patch-size = 33, MSE, SGD, lr = 0.01, epoch = 1000, decay=500
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, scale_factor, args):
        super(Net, self).__init__()

        base_filter = 64
        num_channels = 3
        self.head = ConvBlock(num_channels, base_filter, 9, 1, 4, activation='relu', norm=None, bias = True)

        self.body = ConvBlock(base_filter, 32, 1, 1, 0, activation='relu', norm=None, bias = True)

        self.output_conv = ConvBlock(32, num_channels, 5, 1, 2, activation='relu', norm=None, bias = True)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    # torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    # torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def forward(self, x):

        x = self.head(x)
        x = self.body(x)
        x = self.output_conv(x)

        return x    