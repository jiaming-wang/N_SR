#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-01-29 17:54:45
LastEditTime: 2021-08-20 23:54:08
@Description: batch_size=16, patch_size=32, L1 loss, lr=1e-4, epoch=5000, decay=2500, ADAM
official: batch_size=16, patch_size=32, L1 loss, lr=1e-4, epoch=1000, decay=200, ADAM
'''

import os
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
# from torchvision.transforms import *
import torch

class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        G0_ = G0
        convs = []
        for i in range(C):
            convs.append(DenseBlock_rdn(G0_, G, 3, 1, 1, activation='relu', norm=None))
            G0_ = G0_ + G
        self.dense_layer = nn.Sequential(*convs)
        self.conv_1x1 = nn.Conv2d(G0_, G0, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x_out = self.dense_layer(x)
        x_out = self.conv_1x1(x_out)
        x = x_out + x
        return x

class Net(nn.Module):
    def __init__(self, args):
    # channels, denselayer, growthrate):
        super(Net, self).__init__()
        '''
        D: RDB number 20
        C: the number of conv layer in RDB 6
        G: the growth rate 32
        G0: local and global feature fusion layers 64filter
        '''

        self.args = args
        num_channels = self.args['data']['batch_size']
        scale_factor = self.args['data']['upsacle']
        
        self.D = 20
        self.C = 6
        self.G = 32
        self.G0 = 64

        self.sfe1 = ConvBlock(num_channels, self.G0, 3, 1, 1, activation='relu', norm=None)
        self.sfe2 = ConvBlock(self.G0, self.G0, 3, 1, 1, activation='relu', norm=None)

        self.RDB1 = RDB(self.G0, self.C, self.G)
        self.RDB2 = RDB(self.G0, self.C, self.G)
        self.RDB3 = RDB(self.G0, self.C, self.G)

        self.GFF_1x1 = nn.Conv2d(self.G0*3, self.G0, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(self.G0, self.G0, kernel_size=1, padding=0, bias=True)

        self.up = Upsampler(scale_factor, self.G0, activation=None)
        self.output_conv = ConvBlock(self.G0, num_channels, 3, 1, 1, activation=None, norm=None)

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
        
        x = self.sfe1(x)
        x_0 = self.sfe2(x)
        x_1 = self.RDB1(x_0)
        x_2 = self.RDB2(x_1)
        x_3 = self.RDB3(x_2)
        xx = torch.cat((x_1, x_2, x_3), 1)
        x_LF = self.GFF_1x1(xx)
        x_GF = self.GFF_3x3(x_LF)
        x = x_GF + x
        x = self.up(x)
        x = self.output_conv(x)
        
        return x

class DenseBlock_rdn(torch.nn.Module):
    def __init__(self, in_filter, out_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(DenseBlock_rdn, self).__init__()

        self.conv = torch.nn.Conv2d(in_filter, out_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm

        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            out = self.act(out)
        else:
            out = out
        out = torch.cat((x, out), 1)
        return out