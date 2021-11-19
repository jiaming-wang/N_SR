#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-23 14:57:22
LastEditTime: 2021-11-19 11:44:32
@Description: file content
'''
import torch.utils.data as data
import torch, random, os, math
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
import torch.nn as nn

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img_tar, patch_size, scale, ix=-1, iy=-1):

    (th, tw) = img_tar.size
    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size

    if ix == -1:
        ix = random.randrange(0, tw - tp + 1)
    if iy == -1:
        iy = random.randrange(0, th - tp + 1)

    img_tar = img_tar.crop((iy, ix, iy + tp, ix + tp))
      
    info_patch = {
        'ix': ix, 'iy': iy, 'tp': tp}

    return img_tar, info_patch

def augment(img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True
            
    return img_tar, info_aug

class Data(data.Dataset):
    def __init__(self, image_dir, upscale_factor, cfg, transform=None):
        super(Data, self).__init__()
    
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg
        self.bicubic = bicubic()

    def __getitem__(self, index):
    
        target = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        target = target.crop((0, 0, target.size[0] // self.upscale_factor * self.upscale_factor, target.size[1] // self.upscale_factor * self.upscale_factor))
        target, _ = get_patch(target,self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            target, _ = augment(target)
        
        if self.transform:
            target = self.transform(target)

        C, H, W = target.size()
        input = target.view(-1, C, H, W)

        if self.cfg['data']['blur']:
            self.gen_kernel = Gaussin_Kernel(kernel_size=21, blur_type=self.cfg['data']['blur_type'], sig=2.6, sig_min=0.2, sig_max=4.0,
                lambda_1=0.2, lambda_2=4.0, theta=0, lambda_min=0.2, lambda_max=4.0)
            self.blur = BatchBlur(kernel_size=21)
            
            b_kernels = self.gen_kernel(1, random)
            input = self.blur(input, b_kernels)
            input = input.view(-1, C, H, W)  # BN, C, H, W

        input = self.bicubic(input, scale=1/self.upscale_factor)

        if self.cfg['data']['noise'] != 0:
            noise = torch.randn_like(input).mul_(self.cfg['data']['noise']/255).float().view(-1, input.shape[1], input.shape[2], input.shape[3])
            input = input + noise

        bicubic = self.bicubic(input, scale=self.upscale_factor)
        
        input = torch.clamp(input, 0, 1)
        bicubic = torch.clamp(bicubic, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        input = input.view(input.shape[1], input.shape[2],input.shape[3])  # BN, C, H, W
        bicubic = bicubic.view(bicubic.shape[1], bicubic.shape[2],bicubic.shape[3])  # BN, C, H, W
        # target = target.view(target.shape[1], target.shape[2],target.shape[3])  # BN, C, H, W
        # print(input)
        if self.normalize:
            input = input * 2 - 1
            bicubic = bicubic * 2 - 1
            target = target * 2 - 1

        return input, target, bicubic

    def __len__(self):
        return len(self.image_filenames)

class Data_test(data.Dataset):
    def __init__(self, image_dir, upscale_factor, cfg, transform=None):
        super(Data_test, self).__init__()
        
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg
        self.bicubic = bicubic()

    def __getitem__(self, index):
    
        target = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        target = target.crop((0, 0, target.size[0] // self.upscale_factor * self.upscale_factor, target.size[1] // self.upscale_factor * self.upscale_factor))
        #target, _ = get_patch(target,self.patch_size, self.upscale_factor)
        
        if self.transform:
            target = self.transform(target)
        
        C, H, W = target.size()
        input = target.view(-1, C, H, W)
        
        if self.cfg['data']['blur']:
            self.gen_kernel = Gaussin_Kernel(kernel_size=21, blur_type=self.cfg['data']['blur_type'], sig=2.6, sig_min=0.2, sig_max=4.0,
                lambda_1=0.2, lambda_2=4.0, theta=0, lambda_min=0.2, lambda_max=4.0)
            self.blur = BatchBlur(kernel_size=21)
            C, H, W = target.size()
            b_kernels = self.gen_kernel(1, random)
            input = self.blur(target.view(1, -1, H, W), b_kernels)
            input = input.view(-1, C, H, W)  # BN, C, H, W

        input = self.bicubic(input, scale=1/self.upscale_factor)

        if self.cfg['data']['noise'] != 0:
            noise = torch.randn_like(input).mul_(self.cfg['data']['noise']/255).float().view(-1, input.shape[1], input.shape[2], input.shape[3])
            input = input + noise

        bicubic = self.bicubic(input, scale=self.upscale_factor)

        input = torch.clamp(input, 0, 1)
        bicubic = torch.clamp(bicubic, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        input = input.view(input.shape[1], input.shape[2],input.shape[3])  # BN, C, H, W
        bicubic = bicubic.view(bicubic.shape[1], bicubic.shape[2],bicubic.shape[3])  # BN, C, H, W

        if self.normalize:
            input = input * 2 - 1
            bicubic = bicubic * 2 - 1
            target = target * 2 - 1
            
        return input, target, bicubic, file

    def __len__(self):
        return len(self.image_filenames)

class Data_eval(data.Dataset):
    def __init__(self, image_dir, upscale_factor, cfg, transform=None):
        super(Data_eval, self).__init__()
        
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.normalize = cfg['data']['normalize']

    def __getitem__(self, index):
    
        input = load_img(self.image_filenames[index])      
        bicubic = rescale_img(input, self.upscale_factor)
        _, file = os.path.split(self.image_filenames[index])
           
        if self.transform:
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            
        if self.normalize:
            input = input * 2 - 1
            bicubic = bicubic * 2 - 1
            
        return input, bicubic, file

    def __len__(self):
        return len(self.image_filenames)

######################################
#           define kernel
######################################
class Gaussin_Kernel(object):
    def __init__(self, kernel_size=21, blur_type='iso_gaussian',
                 sig=2.6, sig_min=0.2, sig_max=4.0,
                 lambda_1=0.2, lambda_2=4.0, theta=0, lambda_min=0.2, lambda_max=4.0):
        self.kernel_size = kernel_size
        self.blur_type = blur_type

        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.theta = theta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def __call__(self, batch, random):
        # random kernel
        if random == True:
            return random_gaussian_kernel(batch, kernel_size=self.kernel_size, blur_type=self.blur_type,
                                          sig_min=self.sig_min, sig_max=self.sig_max,
                                          lambda_min=self.lambda_min, lambda_max=self.lambda_max)

        # stable kernel
        else:
            return stable_gaussian_kernel(kernel_size=self.kernel_size, blur_type=self.blur_type,
                                          sig=self.sig,
                                          lambda_1=self.lambda_1, lambda_2=self.lambda_2, theta=self.theta)

def cal_sigma(sig_x, sig_y, radians):
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)

    D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
    U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                   torch.cat([radians.sin(), radians.cos()], 2)], 1)
    sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))

    return sigma


def anisotropic_gaussian_kernel(batch, kernel_size, covar):
    ax = torch.arange(kernel_size).float() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    xy = torch.stack([xx, yy], -1).view(batch, -1, 2)

    inverse_sigma = torch.inverse(covar)
    kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, kernel_size, kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)


def isotropic_gaussian_kernel(batch, kernel_size, sigma):
    ax = torch.arange(kernel_size).float() - kernel_size//2
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

    return kernel / kernel.sum([1,2], keepdim=True)


def random_anisotropic_gaussian_kernel(batch=1, kernel_size=21, lambda_min=0.2, lambda_max=4.0):
    theta = torch.rand(batch) / 180 * math.pi
    lambda_1 = torch.rand(batch) * (lambda_max - lambda_min) + lambda_min
    lambda_2 = torch.rand(batch) * (lambda_max - lambda_min) + lambda_min

    covar = cal_sigma(lambda_1, lambda_2, theta)
    kernel = anisotropic_gaussian_kernel(batch, kernel_size, covar)
    return kernel


def stable_anisotropic_gaussian_kernel(kernel_size=21, theta=0, lambda_1=0.2, lambda_2=4.0):
    theta = torch.ones(1) * theta / 180 * math.pi
    lambda_1 = torch.ones(1) * lambda_1
    lambda_2 = torch.ones(1) * lambda_2

    covar = cal_sigma(lambda_1, lambda_2, theta)
    kernel = anisotropic_gaussian_kernel(1, kernel_size, covar)
    return kernel


def random_isotropic_gaussian_kernel(batch=1, kernel_size=21, sig_min=0.2, sig_max=4.0):
    x = torch.rand(batch) * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(batch, kernel_size, x)
    return k


def stable_isotropic_gaussian_kernel(kernel_size=21, sig=4.0):
    x = torch.ones(1) * sig
    k = isotropic_gaussian_kernel(1, kernel_size, x)
    return k


def random_gaussian_kernel(batch, kernel_size=21, blur_type='iso_gaussian', sig_min=0.2, sig_max=4.0, lambda_min=0.2, lambda_max=4.0):
    if blur_type == 'iso_gaussian':
        return random_isotropic_gaussian_kernel(batch=batch, kernel_size=kernel_size, sig_min=sig_min, sig_max=sig_max)
    elif blur_type == 'aniso_gaussian':
        return random_anisotropic_gaussian_kernel(batch=batch, kernel_size=kernel_size, lambda_min=lambda_min, lambda_max=lambda_max)


def stable_gaussian_kernel(kernel_size=21, blur_type='iso_gaussian', sig=2.6, lambda_1=0.2, lambda_2=4.0, theta=0):
    if blur_type == 'iso_gaussian':
        return stable_isotropic_gaussian_kernel(kernel_size=kernel_size, sig=sig)
    elif blur_type == 'aniso_gaussian':
        return stable_anisotropic_gaussian_kernel(kernel_size=kernel_size, lambda_1=lambda_1, lambda_2=lambda_2, theta=theta)

class BatchBlur(nn.Module):
    def __init__(self, kernel_size=21):
        super(BatchBlur, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        else:
            self.pad = nn.ReflectionPad2d((kernel_size//2, kernel_size//2-1, kernel_size//2, kernel_size//2-1))

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        input_pad = self.pad(input)
        H_p, W_p = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
            kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
        else:
            input_CBHW = input_pad.view((1, C * B, H_p, W_p))
            kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
            kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, groups=B*C).view((B, C, H, W))


######################################
#           define Bicubic
######################################

# implementation of matlab bicubic interpolation in pytorch
class bicubic(nn.Module):
    def __init__(self):
        super(bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0), torch.FloatTensor([in_size[0]])).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1), torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1/4):
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0]
        weight1 = weight1[0]

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out
