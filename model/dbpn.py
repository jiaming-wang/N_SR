#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-01-29 18:19:22
@LastEditTime : 2020-02-16 15:47:39
@Description: batch_size=16, patch_size=32, L1 loss, epoch=2000
'''
import os
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, num_stages, scale_factor, args):
        super(Net, self).__init__()
        self.args = args
        num_stages = 7
        if self.args['data']['upsacle'] == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif self.args['data']['upsacle'] == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif self.args['data']['upsacle'] == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        feat = 256
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
        #Reconstruction
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        self.sub_mean = MeanShift(args['data']['rgb_range'])
        self.add_mean = MeanShift(args['data']['rgb_range'], sign=1)       
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)
        
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        
        concat_h = torch.cat((h2, h1),1)
        l = self.down2(concat_h)
        
        concat_l = torch.cat((l, l1),1)
        h = self.up3(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down3(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up4(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down4(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up5(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down5(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up6(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down6(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up7(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        x = self.output_conv(concat_h)
        
        return x