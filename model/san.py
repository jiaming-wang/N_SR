#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-06-13 22:18:17
LastEditTime: 2021-08-20 23:53:35
@Description: batch_size=16, patch_size=48, L1 loss, epoch=1000, lr=1e-4, decay=200
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
import torch.nn.functional as F

## Second-order Channel Attention Network (SAN)
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        self.args = args
        num_channels = self.args['data']['batch_size']
        scale_factor = self.args['data']['upsacle']
        
        n_resgroups = 20
        n_resblocks = 10
        n_feats = 64
        kernel_size = 3
        reduction = 16
        res_scale = 1
        base_filter = 64
        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        self.head = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='relu', norm=None)

        # define body module
        ## share-source skip connection

        ##
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        self.n_resgroups = n_resgroups
        self.RG = nn.ModuleList([LSRAG(n_feats, kernel_size, reduction, \
                                              res_scale=res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)])
        # self.conv_last = conv(n_feats, n_feats, kernel_size)

        # modules_body = [
        #     ResidualGroup(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups)]
        # modules_body.append(conv(n_feats, n_feats, kernel_size))


        self.up = Upsampler(scale_factor, base_filter, activation=None)
        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation='relu', norm=None)

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.non_local = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats//8, reduction=8,sub_sample=False, bn_layer=False)


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        ## add nonlocal
        xx = self.non_local(x)

        # share-source skip connection
        residual = xx

        # res = self.RG(xx)
        # res = res + xx
        ## share-source residual gruop
        for i,l in enumerate(self.RG):
            xx = l(xx) + self.gamma*residual
            # xx = self.gamma*xx + residual
        # body part
        # res = self.body(xx)
        ##
        ## add nonlocal
        res = self.non_local(xx)
        ##
        # res = self.soca(res)
        # res += x
        res = res + x

        x = self.up(res)
        x = self.output_conv(x)
        # x = self.add_mean(x)

        return x 

## non_local module
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            # sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        # self.g1 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                  kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None
        # self.fc = nn.Linear(64,2304,bias=True)
        # self.sub_bilinear = nn.Upsample(size=(48,48),mode='bilinear')
        # self.sub_maxpool = nn.AdaptiveMaxPool2d(output_size=(48,48))
        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        # output = self.operation_function(x)

        batch_size,C,H,W = x.shape

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # return f
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        output = W_y + x

        return output

    def _embedded_gaussian(self, x):
        batch_size,C,H,W = x.shape
        
        # x_sub = self.sub_bilinear(x) # bilinear downsample
        # x_sub = self.sub_maxpool(x) # maxpool downsample

        ##
        # g_x = x.view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)
        #
        # # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        # theta_x = x.view(batch_size, self.inter_channels, -1)
        # theta_x = theta_x.permute(0, 2, 1)
        # fc = self.fc(theta_x)
        # # phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # # f = torch.matmul(theta_x, phi_x)
        # # return f
        # # f_div_C = F.softmax(fc, dim=-1)
        # return fc

        ##
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        # if self.dimension == 3:
        #     conv_nd = nn.Conv3d
        # elif self.dimension == 2:
        #     conv_nd = nn.Conv2d
        # else:
        #     conv_nd = nn.Conv1d
        # self.g1 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                  kernel_size=1, stride=1, padding=0).to('cuda:0')

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        # g_x = g_x.view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # return f
        f_div_C = F.softmax(f, dim=-1)
        # return f_div_C
        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        # print(f_div_C.shape)
        # print(g_x.shape)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                # nn.Sigmoid()
                # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        _,_,h,w = x.shape
        y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_ave = self.conv_du(y_ave)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_ave

## second-order Channel attention (SOCA)
class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        # subsample
        # subsample_scale = 2
        # subsample = nn.Upsample(size=(h // subsample_scale, w // subsample_scale), mode='nearest')
        # x_sub = subsample(x)
        # max_pool = nn.MaxPool2d(kernel_size=2)
        # max_pool = nn.AvgPool2d(kernel_size=2)
        # x_sub = self.max_pool(x)
        ##
        ## MPN-COV
        cov_mat = CovpoolLayer(x_sub) # Global Covariance pooling layer
        cov_mat_sqrt = SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
        # y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_cov = self.conv_du(cov_mat_sum)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_cov*x

## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8,sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # second-order channel attention
        self.soca=SOCA(in_feat, reduction=reduction)
        # nonlocal module
        self.non_local = (NONLocalBlock2D(in_channels=in_feat,inter_channels=inter_feat, sub_sample=sub_sample,bn_layer=bn_layer))

        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        ## divide feature map into 4 part
        batch_size,C,H,W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]


        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return  nonlocal_feat


## Residual  Block (RB)
class RB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1, dilation=2):
        super(RB, self).__init__()
        modules_body = []

        # self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma1 = 1.0
        # self.salayer = SALayer(n_feat, reduction=reduction, dilation=dilation)
        # self.salayer = SALayer2(n_feat, reduction=reduction, dilation=dilation)



        self.conv_first = nn.Sequential(ConvBlock(n_feat, n_feat, bias=bias, activation='relu'),
                                        ConvBlock(n_feat, n_feat, bias=bias, activation=None)
                                        )


        self.res_scale = res_scale

    def forward(self, x):
        y = self.conv_first(x)
        y = y + x

        return y

## Local-source Residual Attention Group (LSRARG)
class LSRAG(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, res_scale, n_resblocks):
        super(LSRAG, self).__init__()
        ##
        self.rcab= nn.ModuleList([RB(n_feat, kernel_size, reduction, \
                                       bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1) for _ in range(n_resblocks)])
        self.soca = (SOCA(n_feat,reduction=reduction))
        self.conv_last = ConvBlock(n_feat, n_feat, bias=False, activation=None)
        self.n_resblocks = n_resblocks
        ##
        # modules_body = []
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        # for i in range(n_resblocks):
        #     modules_body.append(RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1))
        # modules_body.append(SOCA(n_feat,reduction=reduction))
        # # modules_body.append(Nonlocal_CA(in_feat=n_feat, inter_feat=n_feat//8, reduction =reduction, sub_sample=False, bn_layer=False))
        # modules_body.append(conv(n_feat, n_feat, kernel_size))
        # self.body = nn.Sequential(*modules_body)
        ##

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        # batch_size,C,H,W = x.shape
        # y_pre = self.body(x)
        # y_pre = y_pre + x
        # return y_pre

        ## share-source skip connection

        for i,l in enumerate(self.rcab):
            # x = l(x) + self.gamma*residual
            x = l(x)
        x = self.soca(x)
        x = self.conv_last(x)

        x = x + residual

        return x
        ##


####MPNCOV

import torch
import numpy as np
from torch.autograd import Function

class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input

class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
               Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
               Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            ZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
         y = ZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                          Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) - 
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) - 
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) - 
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag()
         return grad_input, None

class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().t().reshape(dim*dim)
         index = I.nonzero()
         y = torch.zeros(batchSize,dim*(dim+1)/2,device = x.device)
         for i in range(batchSize):
            y[i, :] = x[i, index].t()
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         grad_input = torch.zeros(batchSize,dim,dim,device = x.device,requires_grad=False)
         grad_input = grad_input.reshape(batchSize,dim*dim)
         for i in range(batchSize):
            grad_input[i,index] = grad_output[i,:].reshape(index.size(),1)
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

def TriuvecLayer(var):
    return Triuvec.apply(var)