#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-23 14:57:22
LastEditTime: 2020-11-10 15:34:42
@Description: file content
'''
import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #img = Image.open(filepath)
    #y, _, _ = img.split()
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img_in, img_tar, img_bic, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))
    img_bic = img_bic.crop((ty,tx,ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_bic, info_patch

def augment(img_in, img_tar, img_bic, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_tar, img_bic, info_aug

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

    def __getitem__(self, index):
    
        target = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        target = target.crop((0, 0, target.size[0] // self.upscale_factor * self.upscale_factor, target.size[1] // self.upscale_factor * self.upscale_factor))
        input = target.resize((int(target.size[0]/self.upscale_factor),int(target.size[1]/self.upscale_factor)), Image.BICUBIC)       
        bicubic = rescale_img(input, self.upscale_factor)
        input, target, bicubic, _ = get_patch(input,target,bicubic,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, bicubic, _ = augment(input, target, bicubic)
        
        if self.transform:
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            target = self.transform(target)

        if self.cfg['data']['noise'] != 0:
            noise = torch.randn(B,C,H,W).mul_(self.cfg['data']['noise']).float()
            input = input + noise

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

    def __getitem__(self, index):
    
        target = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        target = target.crop((0, 0, target.size[0] // self.upscale_factor * self.upscale_factor, target.size[1] // self.upscale_factor * self.upscale_factor))
        input = target.resize((int(target.size[0]/self.upscale_factor),int(target.size[1]/self.upscale_factor)), Image.BICUBIC)       
        bicubic = rescale_img(input, self.upscale_factor)
           
        if self.transform:
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            target = self.transform(target)

        if self.cfg['data']['noise'] != 0:
            noise = torch.randn(B,C,H,W).mul_(self.cfg['data']['noise']).float()
            input = input + noise

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
            target = target * 2 - 1
            
        return input, bicubic, file

    def __len__(self):
        return len(self.image_filenames)

def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x