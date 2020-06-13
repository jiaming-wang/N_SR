#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-14 22:38:14
@LastEditTime : 2020-02-16 14:03:44
@Description: batch_size=16, patch_size=24, MSE loss, epoch=8000
'''

import os
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
from model.base_net import *

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, scale_factor, args):
        super(Net, self).__init__()
        self.args = args

        base_filter = 64
        n_resblocks = 16
        #The input layer
        self.feat0 = ConvBlock(num_channels, base_filter, 9, 1, 4, activation='prelu', norm=None)

        body = [
            ResnetBlock(base_filter, 3, 1, 1, activation='prelu', norm=None) for _ in range(n_resblocks)
        ]
        self.body = nn.Sequential(*body)

        self.res_b1 = ConvBlock(base_filter, base_filter, 3, 1, 1, norm=None)

        self.up = Upsampler(scale_factor, base_filter, activation='prelu') 
        #Reconstruction
        self.output_conv = ConvBlock(base_filter, num_channels, 9, 1, 4, activation=None, norm=None)

        self.sub_mean = MeanShift(args['data']['rgb_range'])
        self.add_mean = MeanShift(args['data']['rgb_range'], sign=1) 

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    # torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    # torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.feat0(x)
        stage_output = x
        x = self.body(x)
        x = self.res_b1(x)
        x = x + stage_output
        x = self.up(x)
        x = self.output_conv(x)
        # x = self.add_mean(x)
        return x
