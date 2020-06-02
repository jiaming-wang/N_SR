#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-22 09:46:46
@LastEditTime: 2020-06-02 21:32:41
@Description: batch_size=16, patch_size=48, L1 loss, epoch=300
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import ConvBlock, ResnetBlock_scale, Upsampler, MeanShift
from torchvision.transforms import *

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, num_stages, scale_factor, args):
        super(Net, self).__init__()

        base_filter = 256
        self.sub_mean = MeanShift(args['data']['rgb_range'])
        self.add_mean = MeanShift(args['data']['rgb_range'], sign=1)

        self.head = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='relu', norm=None)

        body = [
            ResnetBlock_scale(base_filter, 0.1, 3, 1, 1, activation='relu', norm=None) for _ in range(32)
        ]

        body.append(ConvBlock(base_filter, base_filter, 3, 1, 1, activation='relu', norm=None))

        self.up = Upsampler(4, base_filter, activation=None)
        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation='relu', norm=None)
        self.body = nn.Sequential(*body)

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
        x = self.head(x)
        res = x
        x = self.body(x)
        x =  res + x

        x = self.up(x)
        x = self.output_conv(x)
        #x = self.add_mean(x)

        return x    

# if __name__ == '__main__':
#     net = Net(3, 64, 256, 4, args=None)
#     x = torch.randn(1, 3, 16, 16)
#     output = net(x)
#     print('Model parameters: '+ str(sum(param.numel() for param in net.params)))
