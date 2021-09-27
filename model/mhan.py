#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-01-29 18:19:22
LastEditTime: 2021-08-22 10:48:52
@Description: batch_size=48, patch_size=64, L1 loss, lr=1e-4, epoch=2000, ADAM, decay=1000
'''
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import math

class HighDivModule(nn.Module):
    def __init__(self, in_channels, order=3):
        super(HighDivModule, self).__init__()
        self.order = order
        self.inter_channels = in_channels // 8 * 2
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, padding=0, bias=False))
            )
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            setattr(self, name, nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 1, padding=0, bias=False),
                                              nn.Sigmoid()
                                   )
                                   )

    def forward(self, x):
        y=[]
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                layer = getattr(self, name)
                y.append(layer(x))
        y_ = []
        cnt=0
        for j in range(self.order):
            y_temp = 1
            for i in range(j+1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(F.relu(y_temp))
        
        #y_ = F.relu(y_)
        y__ = 0
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            layer = getattr(self, name)
            y__ += layer(y_[i])
        out = x * y__ / self.order
        return out#, y__/ self.order

class HighDivBlock(nn.Module):
    def __init__(self, features):
        super(HighDivBlock, self).__init__()
        self.conv_1 = nn.Conv2d(features*3, features, kernel_size=1, bias=True)
        self.conv_2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.HID = HighDivModule(features, 3)
        self.relu = torch.nn.LeakyReLU(0.2, True)
    def forward(self, x1, x2, x3):
        x = torch.cat((x1,x2,x3), 1)
        out = self.relu(self.conv_1(x))
        out = self.HID(out)
        out = self.relu(self.conv_2(out))
        return out

class KernelPredeictionModule(nn.Module):

    def __init__(self, input_channel, channel_cm=64, kernel_up=5, kernel_encoder=3, enlarge_rate=2):
        super(KernelPredeictionModule,self).__init__()
        self.input_channel = input_channel
        self.channel_cm = channel_cm
        self.kernel_up = kernel_up
        self.kernel_encoder = kernel_encoder
        self.enlarge_rate = enlarge_rate
        self.channel_compressor = nn.Sequential(
            OrderedDict([
                ("compressor_conv" , nn.Conv2d(self.input_channel, self.channel_cm,1,1,0,bias=False)),
               # ("compressor_bn"   , nn.BatchNorm2d(self.channel_cm)),
                ("compressor_relu" , nn.ReLU(inplace=True))
            ])
        )
        self.context_encoder = nn.Sequential(
            OrderedDict([
                ("encoder_conv"    , nn.Conv2d(self.channel_cm,
                                          self.enlarge_rate*self.enlarge_rate*self.kernel_up*self.kernel_up,# rate^2*kup^2
                                          self.kernel_encoder,padding=int((self.kernel_encoder-1)/2),bias=False)),
                #("encoder_bn"      , nn.BatchNorm2d(self.enlarge_rate*self.enlarge_rate*self.kernel_up*self.kernel_up)),
                ("encoder_relu"    , nn.ReLU(inplace=True))
            ])
        )
        self.kernel_normalizer = nn.Softmax(dim=-1)
    def forward(self, x):
        b,c,w,h = x.shape
        x = self.channel_compressor(x)
        x = self.context_encoder(x)
        x = x.view(b,self.kernel_up*self.kernel_up,self.enlarge_rate*w,self.enlarge_rate*h)# batch*(kup^2)*(rate*w)*(rate*h)
        x = self.kernel_normalizer(x)
        #print("KP cost:{}".format(datetime.datetime.now() - start_time))
        return x

class Carafe(nn.Module):
    def __init__(self, input_channel, channel_cm=64, kernel_up=5, kernel_encoder=3, enlarge_rate=2):
        """
        The Carafe upsample model(unoffical)
        :param input_channel: The channel of input
        :param channel_cm:    The channel of Cm, paper give this parameter 64
        :param kernel_up:     The kernel up, paper give this parameter 5
        :param kernel_encoder:The kernel encoder, paper suggest it kernel_up-2, so 3 here
        :param enlarge_rate:  The enlarge rate , your rate for upsample (2x usually)
        """
        super(Carafe, self).__init__()
        self.kernel_up = kernel_up
        self.enlarge_rate = enlarge_rate
        self.KPModule = KernelPredeictionModule(input_channel,channel_cm,kernel_up,kernel_encoder,enlarge_rate)

    def forward(self, x):

        # KernelPredeictionModule : cost 0.7175s
        kpresult = self.KPModule(x) # (b,kup*kup,e_w,e_h)


        ############Context-aware Reassembly Module########################
        ######## Step1 formal_pic deal : cost 0.1164s
        x_mat = self.generate_kup_mat(x)

        ######## Step2 kernel deal : cost 0.001s
        channel = x.shape[1]
        w_mat = self.repeat_kernel(kpresult,channel)

        ######## Step3 kernel mul : cost 0.0009s
        output = torch.mul(x_mat,w_mat)

        ######## Step4 sum the kup dim : cost 0.0002s
        output = torch.sum(output, dim=2)
        return output

    def generate_kup_mat(self,x):
        """
        generate the mat matrix, make a new dim kup for mul
        :param x:(batch,channel,w,h)
        :return: (batch,channel,kup*kup,enlarged_w,enlarged_h)
        """
        batch, channel, w ,h = x.shape
        # stride to sample
        r = int(self.kernel_up / 2)
        # get the dim kup**2 with unfold with the stride windows
        x_mat = torch.nn.functional.unfold(x, kernel_size=self.kernel_up, padding=r, stride=1)
        # make the result to (b,c,kup**2,w,h)
        x_mat = x_mat.view((batch, channel, self.kernel_up**2, w, h))
        # nearest inter the number for i map the region [i:i/enlarge,j:j/enlarge]
        x_mat = torch.nn.functional.interpolate(x_mat,
                                                scale_factor=(1, self.enlarge_rate, self.enlarge_rate),
                                                mode='nearest')
        #print("inter cost:{}".format(datetime.datetime.now() - start_time))
        return x_mat

    def repeat_kernel(self,weight,channel):
        """
        Generate the channel dim for the weight
        repeat the Kernel Prediction Module output for channel times,
        and it can be mul just like the depth-width conv (The repeat on the batch dim)
        :param weight:  (batch,kup*kup,enlarged_w,enlarged_h)
        :param channel: the channel num to repeat
        :return: (batch,channel,kup*kup,enlarged_w,enlarged_h)
        """
        batch, kup_2, w, h = weight.shape
        # copy the channel in batch
        w_mat = torch.stack([i.expand(channel, kup_2, w, h) for i in weight])
        # each channel in batch is the same!
        # print(torch.equal(w_mat[0, 0, ...], w_mat[0, 1, ...]))
        return w_mat

        
class trans_block(nn.Module):
    def __init__(self, features):
        super(trans_block, self).__init__()

        self.conv_1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv_d1 = nn.Conv2d(features, features, kernel_size=3, stride = 2, padding=1, bias=True)
        self.conv_d2 = nn.Conv2d(features, features, kernel_size=3, stride = 2, padding=1, bias=True)
        self.act = torch.nn.LeakyReLU(0.2, True)
        #self.conv_up1 = upsample(features, stride = 2, mode = mode)
        #self.conv_up2 = upsample(features, stride = 2, mode = mode)
    def channel_shuffle_2D(self, x, groups):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups,
            channels_per_group, height, width)
        #permute
        x = x.permute(0,2,1,3,4).contiguous()
        # flatten
        x = x.view(batchsize, num_channels, height, width)
        return x

    def forward(self, x):
        o1 = self.act(self.conv_1(x))
        o2 = self.act(self.conv_2(o1))
        o3 = self.act(self.conv_3(o2))
        o4 = self.act(self.conv_d1(o1))
        o5 = self.act(self.conv_d2(o4))
        o5 = self.channel_shuffle_2D(o5, 4)
        #o6 = self.conv_up1(o5)
        o6 = torch.nn.functional.interpolate(o5, o4.shape[2:],mode='bilinear',align_corners=True)
        o6 = o6 + self.act(self.conv_4(o4))
        #o7 = self.conv_up2(o6)
        o7 = torch.nn.functional.interpolate(o6, o3.shape[2:],mode='bilinear',align_corners=True)
        o7 = o7 + self.act(self.conv_5(o3))
        return o7 + x

class cat_block(nn.Module):
    def __init__(self, features):
        super(cat_block, self).__init__()
        self.conv_1 = nn.Conv2d(features, 4 * features, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(4 * features, features, kernel_size=3, padding=1, bias=True)
        self.conv_3 = nn.Conv2d(features * 2, features, kernel_size=3, padding=1, bias=True)
        self.act = torch.nn.LeakyReLU(0.2, True)
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        skip = x
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = torch.cat((self.gamma1 * skip, self.gamma2 * x), 1)
        x = self.conv_3(x)
        return x

class cat_group(nn.Module):
    def __init__(self, features, nUnit):
        super(cat_group, self).__init__()
        modules_body = [
            cat_block(features)
            for _ in range(nUnit)]
        self.body = nn.Sequential(*modules_body)
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        self.conv = nn.Conv2d(features * 2, features, kernel_size=3, padding=1, bias=True)
    def forward(self, x):
        skip = x
        x = self.body(x)
        x = torch.cat((self.gamma1 * skip, self.gamma2 * x), 1)
        x = self.conv(x)
        return x

class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, groups=1):
        super(_UpsampleBlock, self).__init__()

        self.body = nn.ModuleList()
        if scale in [2, 4, 8]:
            for _ in range(int(math.log(scale, 2))):
                self.body.append(
                    nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=groups)
                )
                self.body.append(nn.ReLU(inplace=True))
                self.body.append(nn.PixelShuffle(2))
        elif scale == 3:
            self.body.append(
                nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=groups)
            )
            self.body.append(nn.ReLU(inplace=True))
            self.body.append(nn.PixelShuffle(3))

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return out
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args
        in_channels = self.args['data']['n_colors']
        scale = self.args['data']['upsacle']
        num_features = 8

        self.conv_in = nn.Conv2d(in_channels, num_features * 4, kernel_size=3, padding=1, bias=True)
        self.feat_in = nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1, bias=True)
        self.relu = torch.nn.LeakyReLU(0.2, True)
        nUnit = 3
        self.T_block_1 = cat_group(num_features, nUnit)
        self.T_block_2 = cat_group(num_features, nUnit)
        self.T_block_3 = cat_group(num_features, nUnit)
        self.T_block_4 = cat_group(num_features, nUnit)
        self.T_block_5 = cat_group(num_features, nUnit)
        self.T_block_6 = cat_group(num_features, nUnit)
        self.T_block_7 = cat_group(num_features, nUnit)
        self.T_block_8 = cat_group(num_features, nUnit)
        self.T_block_9 = cat_group(num_features, nUnit)
        
        self.upsample = _UpsampleBlock(num_features, scale)
        #self.upsample = Carafe(input_channel=num_features,channel_cm=32)
        self.HDB_1 = HighDivBlock(num_features)
        self.HDB_2 = HighDivBlock(num_features)
        self.HDB_3 = HighDivBlock(num_features)
        self.HDB_4 = HighDivBlock(num_features)

        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True)
    def forward(self, x):
        fea = self.relu(self.conv_in(x))
        fea_0 = self.relu(self.feat_in(fea))
        
        fea_1 = self.T_block_1(fea_0)
        fea_2 = self.T_block_2(fea_1)
        fea_3 = self.T_block_3(fea_2)
        fea_4 = self.T_block_4(fea_3)
        fea_5 = self.T_block_5(fea_4)
        fea_6 = self.T_block_6(fea_5)
        fea_7 = self.T_block_7(fea_6)
        fea_8 = self.T_block_8(fea_7)
        fea_9 = self.T_block_9(fea_8)
        
        fea_10 = self.HDB_1(fea_9,  fea_5, fea_4)
        fea_11 = self.HDB_2(fea_10, fea_6, fea_3)
        fea_12 = self.HDB_3(fea_11, fea_7, fea_2)
        fea_13 = self.HDB_4(fea_12, fea_8, fea_1)

        
        fea_up = self.upsample(fea_13)
        fea_out = self.conv3(fea_up)
        img_out = self.conv4(fea_out)
        
        return img_out
        
     
# if __name__ == '__main__':       
#     import time 
#     x = torch.randn(1,3,100,100)
#     Net = Net(3, 64, 8).cuda()
#     print_network(Net)
#     x = x.cuda()
#     t0 = time.time()
#     for i in range(30):
#         out = Net(x)
#     t = time.time() - t0
#     print('average running time: ', t/30)

#     print(out.shape)
