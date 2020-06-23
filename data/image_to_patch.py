#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-06-13 20:12:23
@LastEditTime: 2020-06-23 15:00:30
@Description: file content
'''
import numpy as np 
import glob, os, h5py
import os
# import cv2
from scipy import misc

class image_to_patch:
    
    def __init__(self, patch_size, scale, image_path, output_filename):

        self.hr_patch_size = patch_size
        self.lr_patch_size = patch_size
        self.scale = scale
        self.image_path = image_path
        self.labels = []
        self.inputs = []
        self.batch_size = 64
        self.stride = patch_size
        self.output_filename = output_filename

    def to_patch(self):
        
        data = glob.glob(os.path.join(self.image_path, '*.png'))

        for i in range(len(data)):

            img = self.imread(data[i])
            img = self.modcrop(img, self.scale)
            img_lr = misc.imresize(img, 1 / self.scale, interp='bicubic')
            img_bic = misc.imresize(img_lr, float(self.scale), interp='bicubic')
            # misc.imsave('outfile.png', img_bic)
            # print(img_bic.shape)
            img = img.astype(np.float32) / 255
            img_bic = np.clip(img_bic.astype(np.float32) / 255, 0.0, 1.0)

            if len(img.shape) == 3: # is color
                h, w, c = img.shape
            else:
                h, w = img.shape # is grayscale

            for x in range(0, h - self.hr_patch_size, self.stride):
                for y in range(0, w - self.hr_patch_size, self.stride):

                    sub_img_label = img[x: x + self.hr_patch_size, y: y + self.hr_patch_size]
                    sub_img_input = img_bic[x: x + self.hr_patch_size, y: y + self.hr_patch_size]

                    self.labels.append(sub_img_label)
                    self.inputs.append(sub_img_input)
                    
        self.labels = np.array(self.labels)
        self.inputs = np.array(self.inputs)
        self.shuffle()
        self.write_hdf5(self.inputs, self.labels, self.output_filename)

    # def imsave(self, image, path):
 
    #     if not os.path.isdir(os.path.join(os.getcwd())):
    #         os.makedirs(os.path.join(os.getcwd()))

    #     cv2.imwrite(os.path.join(os.getcwd(),path),image)

    def imread(self, path):
        img = misc.imread(path)
        return img
    
    def shuffle(self):
        index = list(range(len(self.labels)))
        np.random.shuffle(index)
        self.labels = self.labels[index]
        self.inputs = self.inputs[index]

    def write_hdf5(self, data, labels, output_filename):

        x = data.astype(np.float32)
        y = labels.astype(np.float32)

        print(x.shape)
        with h5py.File(output_filename, 'w') as h:
            h.create_dataset('data', data=x, shape=x.shape)
            h.create_dataset('label', data=y, shape=y.shape)

    def modcrop(self, img, scale =3):

        if len(img.shape) ==3:
            h, w, _ = img.shape
            h = (h // scale) * scale
            w = (w // scale) * scale
            img = img[0:h, 0:w, :]
        else:
            h, w = img.shape
            h = (h // scale) * scale
            w = (w // scale) * scale
            img = img[0:h, 0:w]
        return img


if __name__ == '__main__':
    csnln_path = 41
    scale = 4
    image_pach_train = r'/home/wjmecho/workdir/SR/N_SR-master/dataset/hr/'
    output_filename_train = 'train.h5'
    train = image_to_patch(csnln_path, scale, image_pach_train, output_filename_train)
    train.to_patch()

    image_pach_test = r'/home/wjmecho/workdir/SR/N_SR-master/dataset/valid/'
    output_filename_test = 'test.h5'
    test = image_to_patch(csnln_path, scale, image_pach_test, output_filename_test)
    test.to_patch()