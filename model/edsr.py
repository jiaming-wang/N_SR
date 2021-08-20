#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-22 09:46:46
LastEditTime: 2021-08-20 23:54:58
@Description: batch_size=16, patch_size=48, L1 loss, epoch=300, ADAM, decay=150, lr=1e-4
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
# from torchvision.transforms import *
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args
        num_channels = self.args['data']['batch_size']
        scale_factor = self.args['data']['upsacle']
        
        base_filter = 256
        n_resblocks = 32

        # self.sub_mean = MeanShift(args['data']['rgb_range'])
        # self.add_mean = MeanShift(args['data']['rgb_range'], sign=1)

        self.head = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='relu', norm=None)

        body = [
            ResnetBlock(base_filter, 3, 1, 1, 0.1, activation='relu', norm=None) for _ in range(n_resblocks)
        ]

        body.append(ConvBlock(base_filter, base_filter, 3, 1, 1, activation='relu', norm=None))

        self.up = Upsampler(scale_factor, base_filter, activation=None)
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

if __name__ == "__main__":
    input = Variable(torch.FloatTensor(1, 3, 40, 40))
    model = Net(num_channels= 3,base_filter=1,scale_factor= 1,args= 1)
    out = model(input)
    print(out.shape)