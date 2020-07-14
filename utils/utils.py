#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:12:52
@LastEditTime: 2020-07-11 20:17:48
@Description: file content
'''
import os, math, torch,cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.vgg import VGG
import torch.nn.functional as F

def maek_optimizer(opt_type, cfg, params):
    if opt_type == "ADAM":
        optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'], betas=(cfg['schedule']['beta1'], cfg['schedule']['beta2']), eps=cfg['schedule']['epsilon'])
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg['schedule']['lr'], momentum=cfg['schedule']['momentum'])
    elif opt_type == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg['schedule']['lr'], alpha=cfg['schedule']['alpha'])
    else:
        raise ValueError
    return optimizer

def make_loss(loss_type):
    if loss_type == "MSE":
        loss = nn.MSELoss()
    elif loss_type == "L1":
        loss = nn.L1Loss()
    elif loss_type == "VGG22":
        loss = VGG(loss_type[3:], rgb_range=255)
    elif loss_type == "VGG54":
        loss = VGG(loss_type[3:], rgb_range=255)
    else:
        raise ValueError
    return loss

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class CycleLoss(nn.Module):
    def __init__(self, CycleLoss_weigth = 1, scale = 1/2, loss_type = 'L1'):
        
        self.CycleLoss_weigth = CycleLoss_weigth
        self.scale = scale

        if loss_type == "MSE":
            self.loss = nn.MSELoss()
        elif loss_type == "L1":
            self.loss = nn.L1Loss()
        else:
            raise ValueError

    def forward(self, x_hr, x_lr):
        down_x = F.interpolate(x_hr, scale_factor=self.scale, mode='bicubic')
        return self.loss(down_x, x_lr)
        
def get_path(subdir):
    return os.path.join(subdir)

def save_config(time, log):
    open_type = 'a' if os.path.exists(get_path('./log/' + str(time) + '/records.txt'))else 'w'
    log_file = open(get_path('./log/' + str(time) + '/records.txt'), open_type)
    log_file.write(str(log) + '\n')

def save_net_config(time, log):
    open_type = 'a' if os.path.exists(get_path('./log/' + str(time) + '/net.txt'))else 'w'
    log_file = open(get_path('./log/' + str(time) + '/net.txt'), open_type)
    log_file.write(str(log) + '\n')

def draw_curve_and_save(x, y, title, filename, precision):
    if not isinstance(x, np.ndarray):
        x = np.array(x).astype(np.int32)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_title(title)

    max_y = np.ceil(y.max() / precision) * precision
    min_y = np.floor(y.min() / precision) * precision
    major_y_step = (max_y - min_y) / 10
    if major_y_step < 0.1:
        major_y_step = 0.1
    #设置时间间隔
    ax.yaxis.set_major_locator(MultipleLocator(major_y_step))
    #设置副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(major_y_step))
    ax.yaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='both')
    # ax.legend()
    if (x.shape[0] >= 2):
        axis_range = [x.min(), x.man(), min_y, max_y]
        ax.axis(axis_range)
    ax.plot(x, y)
    plt.savefig(filename)

def calculate_psnr(img1, img2, pixel_range=255, color_mode='rgb'):
    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2
    # img1 and img2 have range [0, pixel_range]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(pixel_range / math.sqrt(mse))

def ssim(img1, img2, pixel_range=255, color_mode='rgb'):
    C1 = (0.01 * pixel_range)**2
    C2 = (0.03 * pixel_range)**2

    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, pixel_range=255):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, pixel_range)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, pixel_range))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), pixel_range)
    else:
        raise ValueError('Wrong input image dimensions.')