#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-07-26 00:26:20
LastEditTime: 2021-08-20 23:47:35
Description: file content
'''

import matplotlib.pyplot as plt
import numpy as np

def smoothing(data, weight):
    last = data[0]
    new_data = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        new_data.append(smoothed_val)
        last = smoothed_val
    return np.array(new_data)
    
def read_txt(txt_path):
    loss_list = []
    psnr_list = []
    ssim_list = []
    file_obj = open(txt_path)
    all_lines = file_obj.readlines()
    for line in all_lines:
        try:
            if "Loss=" in line:
                loss_value = line.split('Loss=')[1].replace('\n','')
                loss_list.append(float(loss_value))
            elif "PSNR=" in line:
                psnr_value = line.split('PSNR=')[1]
                psnr_list.append(float(psnr_value[0:7]))
            elif "SSIM=" in line:
                ssim_value = line.split('SSIM=')[1].replace('\n','')
                ssim_list.append(float(ssim_value))
        finally:
            pass
    file_obj.close()
    return psnr_list, loss_list, ssim_list

def list2curve(loss_list):
    index = [i for i in range(0,120)]
    plt.plot(index, loss_list[0:120], label='1')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.show()
