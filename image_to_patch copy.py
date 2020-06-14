#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-06-13 20:12:23
@LastEditTime: 2020-06-13 21:19:24
@Description: file content
'''
import numpy as np 
import glob
import os
import cv2

class image_to_patch:
    
    def __init__(self, patch_size, scale, image_patch):

        self.hr_patch_size = patch_size * scale
        self.lr_patch_size = patch_size
        self.scale = scale
        self.image_patch = image_patch

    def to_patch(self):
        
        data = glob.glob(os.path.join(self.image_patch, '*.png'))

        for i in range(len(data)):

            img = self.imread(data[i])
            if len(img.shape) == 3: # is color
                h, w, c = img.shape
            else:
                h, w = img.shape # is grayscale
            nx, ny = 0, 0

            image_path = data[i]

            n = 1
            for x in range(0, h - self.hr_patch_size, self.hr_patch_size):
                nx += 1; ny = 0
                for y in range(0, w - self.hr_patch_size, self.hr_patch_size):
                    ny += 1

                    sub_img = img[x: x + self.hr_patch_size, y: y + self.hr_patch_size]

                    self.imsave(sub_img, image_path[:-4] + '_' + str(n) + '.png')
                    n = n + 1

            for x in range(0, h - self.hr_patch_size, self.hr_patch_size):
                nx += 1; ny = 0
                for y in range(w - self.hr_patch_size, w, self.hr_patch_size):
                    ny += 1

                    sub_img = img[x: x + self.hr_patch_size, y: y + self.hr_patch_size]

                    self.imsave(sub_img, image_path[:-4] + '_' + str(n) + '.png')
                    n = n + 1

            for x in range(h - self.hr_patch_size, h, self.hr_patch_size):
                nx += 1; ny = 0
                for y in range(0, w - self.hr_patch_size, self.hr_patch_size):
                    ny += 1

                    sub_img = img[x: x + self.hr_patch_size, y: y + self.hr_patch_size]

                    self.imsave(sub_img, image_path[:-4] + '_' + str(n) + '.png')
                    n = n + 1
            
            for x in range(h - self.hr_patch_size, h, self.hr_patch_size):
                nx += 1; ny = 0
                for y in range(w - self.hr_patch_size, w, self.hr_patch_size):
                    ny += 1

                    sub_img = img[x: x + self.hr_patch_size, y: y + self.hr_patch_size]

                    self.imsave(sub_img, image_path[:-4] + '_' + str(n) + '.png')
                    n = n + 1

    def imsave(self, image, path):
 
        if not os.path.isdir(os.path.join(os.getcwd())):
            os.makedirs(os.path.join(os.getcwd()))

        cv2.imwrite(os.path.join(os.getcwd(),path),image)

    def imread(self, path):
        img = cv2.imread(path)
        return img
    
    def modcrop(self, img, scale =3):
        # Check the image is grayscale
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
    csnln_path = 48
    scale = 4
    image_patch = r'/home/wjmecho/workdir/SR/N_SR-master/dataset/valid'
    myclass = image_to_patch(csnln_path, scale, image_patch)
    myclass.to_patch()