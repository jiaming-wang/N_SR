#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-17 22:19:38
LastEditTime: 2021-08-20 23:44:53
@Description: file content
'''
from solver.basesolver import BaseSolver
import os, torch, time, cv2, importlib
import torch.backends.cudnn as cudnn
from data.data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable 
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)
        
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net
        
        self.model = net(
                args = self.cfg
        )
        self.fmap_block = list()
        self.input_block = list()
    
    ## define hook
    def forward_hook(self, module, data_input, data_output):
        self.fmap_block.append(data_output)
        self.input_block.append(data_input)
        
    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)
            torch.cuda.set_device(self.gpu_ids[0]) 
            
            self.model_path = os.path.join(self.cfg['checkpoint'], self.cfg['test']['model'])

            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
            self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage)['net'])

    def test(self):
        self.model.eval()
        avg_time= []
        for batch in self.data_loader:          
            input, target, bicubic, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), batch[3]
            if self.cuda:
                input = input.cuda(self.gpu_ids[0])
                target = target.cuda(self.gpu_ids[0])
                bicubic = bicubic.cuda(self.gpu_ids[0])

            if self.cfg['algorithm'] == 'VDSR' or self.cfg['algorithm'] == 'SRCNN':
                input = bicubic
            
            ## hook
            # if self.cuda:
            #     hadle_hook = self.model.module.res_b1.register_forward_hook(self.forward_hook)
            # else:
            #     hadle_hook = self.model.res_b1.register_forward_hook(self.forward_hook)

            t0 = time.time()
            with torch.no_grad():
                prediction = self.model(input)
            t1 = time.time()

            if self.cfg['data']['normalize'] :
                target = (target+1) /2
                prediction = (prediction+1) /2
                bicubic = (bicubic+1) /2

            ## remove hook, save feature maps
            # hadle_hook.remove()
            # self.fmap_block = self.fmap_block[0].squeeze().detach().cpu()
            # self.fmap_block = (self.fmap_block*255).numpy().astype(np.uint8)
            # for i in range(0, self.fmap_block[0].shape[1]-1):
            #     plt.imsave('./1/{}.png'.format(str(i)), self.fmap_block[i,:,:], cmap = plt.cm.jet)
            # self.fmap_block = list()
            # self.input_block = list()

            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)
            self.save_img(bicubic.cpu().data, name[0][0:-4]+'_bic.png')
            self.save_img(target.cpu().data, name[0][0:-4]+'_gt.png')
            self.save_img(prediction.cpu().data, name[0][0:-4]+'.png')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))
        
    def eval(self):
        self.model.eval()
        avg_time= []
        for batch in self.data_loader:
            
            input, bicubic, name = Variable(batch[0]), Variable(batch[1]), batch[2]
            if self.cuda:
                input = input.cuda(self.gpu_ids[0])
                bicubic = bicubic.cuda(self.gpu_ids[0])

            t0 = time.time()
            with torch.no_grad():   
                prediction = self.model(input)
            t1 = time.time()
            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)
            self.save_img(bicubic.cpu().data, name[0][0:-4]+'_Bic.png')
            self.save_img(prediction.cpu().data, name[0][0:-4]+'.png')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))

    def save_img(self, img, img_name):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        # save img
        save_dir=os.path.join('results/',self.cfg['test']['type'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = save_dir +'/'+ img_name
        cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
   
    def run(self):
        self.check()
        if self.cfg['test']['type'] == 'test':            
            self.dataset = get_test_data(self.cfg, self.cfg['test']['test_dataset'], self.cfg['data']['upsacle'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                num_workers=self.cfg['threads'])
            self.test()
        elif self.cfg['test']['type'] == 'eval':            
            self.dataset = get_eval_data(self.cfg, self.cfg['test']['test_dataset'], self.cfg['data']['upsacle'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                num_workers=self.cfg['threads'])
            self.eval()
        else:
            raise ValueError('Mode error!')