#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-06-18 20:25:26
LastEditTime: 2020-11-20 09:38:21
@Description: file content
'''
import numpy as np 
import glob, os, math, h5py, cv2
from scipy import misc
# from utils.iqa.vifp import vifp_mscale

class compare_psnr_ssim:
    
    def __init__(self, scale, path, num):

        self.scale = scale
        self.path = path
        self.num = num

        self.psnr_lr = []
        self.psnr_sr = []
        self.ssim_lr = []
        self.ssim_sr = []
        self.fsim_lr = []
        self.fsim_sr = []
        self.vif_lr = []
        self.vif_sr = []
  
    
    def run(self):
        for i in range(1, int(self.num)+1):
            self.hr_path = os.path.join(self.path, str(i) + '_gt.png')
            self.bic_path = os.path.join(self.path, str(i) + '_bic.png')
            self.sr_path = os.path.join(self.path, str(i) + '.png')

            self.hr = self.imread(self.hr_path)
            self.bic = self.imread(self.bic_path)
            self.sr = self.imread(self.sr_path)
            
            pixel_range = 255
            self.hr = (self.hr * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
            self.bic = (self.bic * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
            self.sr = (self.sr * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range

            self.hr = self.hr.astype(np.float64)
            self.bic = self.bic.astype(np.float64)
            self.sr = self.sr.astype(np.float64)
            
            self.psnr_lr.append(self.psnr(self.bic, self.hr))
            self.psnr_sr.append(self.psnr(self.sr, self.hr))

            self.ssim_lr.append(self.ssim(self.bic, self.hr))
            self.ssim_sr.append(self.ssim(self.sr, self.hr))
            
            # self.vif_lr.append(vifp_mscale(self.bic, self.hr))
            # self.vif_sr.append(vifp_mscale(self.sr, self.hr))

        print('The average psnr of LR : ' + str(np.mean(self.psnr_lr)))
        print('The average psnr of SR : ' + str(np.mean(self.psnr_sr)))
        print('The average ssim of LR : ' + str(np.mean(self.ssim_lr)))
        print('The average ssim of SR : ' + str(np.mean(self.ssim_sr)))
        # print('The average vif of LR : ' + str(np.mean(self.vif_lr)))
        # print('The average vif of SR : ' + str(np.mean(self.vif_sr)))

    
    def psnr(self, lr, hr):
        lr = lr[self.scale:-self.scale, self.scale:-self.scale]
        hr = hr[self.scale:-self.scale, self.scale:-self.scale]
        rmse = np.mean((lr - hr)**2)
        return 20 * math.log10(255 / math.sqrt(rmse))

    # def ssim()
    def ssim(self, lr, hr):

        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = lr
        img2 = hr
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

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def imread(self, path, mode=None):

        img = misc.imread(path,mode=mode)
        return img
        
    def modcrop(self, img, scale = 3):

        if len(img.shape) ==3:
            h, w, _ = img.shape
            h = (h / scale) * scale
            w = (w / scale) * scale
            img = img[0:h, 0:w, :]
        else:
            h, w = img.shape
            h = (h / scale) * scale
            w = (w / scale) * scale
            img = img[0:h, 0:w]
        return img


if __name__ == '__main__':
    a = compare_psnr_ssim(4, r'C://Users//Wang//Desktop//test', 10)      
    a.run()
        