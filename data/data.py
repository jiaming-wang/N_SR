#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-16 19:22:41
@LastEditTime: 2020-06-23 14:59:44
@Description: file content
'''
from os.path import join
from torchvision.transforms import Compose, ToTensor
from .dataset import Data, Data_test, Data_eval, Data_patch
from torchvision import transforms
import torch, h5py, numpy
import torch.utils.data as data

def transform():
    return Compose([
        ToTensor(),
    ])
    
def get_data(cfg, data_dir, upscale_factor):
    data_dir = join(cfg['data_dir'], data_dir)
    patch_size = cfg['data']['patch_size']
    augmentation = cfg['data']['data_augmentation']
    normalize = cfg['data']['normalize']
    return Data(data_dir, patch_size, upscale_factor, augmentation, normalize, transform=transform())

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = numpy.array(hf.get('data'))
        self.target = numpy.array(hf.get('label'))
        self.data = numpy.transpose(self.data, (0, 3, 1, 2))
        self.target = numpy.transpose(self.target, (0, 3, 1, 2))

    def __getitem__(self, index):
        # self.data = numpy.transpose(self.data)
        # self.target = numpy.transpose(self.target)
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        
        return self.data.shape[0]
    
def get_test_data(cfg, data_dir, upscale_factor):
    data_dir = join(cfg['test']['data_dir'], data_dir)
    return Data_test(data_dir, upscale_factor, transform=transform())

def get_eval_data(cfg, data_dir, upscale_factor):
    data_dir = join(cfg['test']['data_dir'], data_dir)
    return Data_eval(data_dir, upscale_factor, transform=transform())