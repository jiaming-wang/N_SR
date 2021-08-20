#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:04:48
LastEditTime: 2021-08-20 23:37:51
@Description: file content
'''
import os, importlib, torch, shutil
from torch.cuda.amp import autocast as autocast, GradScaler
from solver.basesolver import BaseSolver
from utils.utils import maek_optimizer, make_loss, calculate_psnr, calculate_ssim, save_config, save_net_config, save_net_py
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from torch.optim import lr_scheduler
from importlib import import_module
from torch.autograd import Variable
from data.data import DatasetFromHdf5
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.config import save_yml
from torchsummary import summary
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.init_epoch = self.cfg['schedule']
        
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        if self.cfg['algorithm'] == 'VDSR' or self.cfg['algorithm'] == 'SRCNN':
            
            train_dataset = DatasetFromHdf5("data/train.h5")
            self.train_loader = DataLoader(train_dataset, cfg['data']['batch_size'], shuffle=False,
                num_workers=1)
            val_dataset = DatasetFromHdf5("data/test.h5")
            self.val_loader = DataLoader(val_dataset, cfg['data']['batch_size'], shuffle=False,
                num_workers=1)

        self.model = net(
            args = self.cfg
        )
        
        self.optimizer = maek_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())
        self.milestones = list(map(lambda x: int(x), self.cfg['schedule']['decay'].split('-')))
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, self.milestones, gamma=self.cfg['schedule']['gamma'], last_epoch=-1)
        self.loss = make_loss(self.cfg['schedule']['loss'])

        self.log_name = self.cfg['algorithm'] + '_' + str(self.cfg['data']['upsacle']) + '_' + str(self.timestamp)
        # save log
        self.writer = SummaryWriter('log/' + str(self.log_name))
        # if not self.cfg['algorithm'] == 'SMSR':
        #     summary(self.model, (3, self.cfg['data']['patch_size'], self.cfg['data']['patch_size']), device='cpu')
        save_net_config(self.log_name, self.model)
        save_net_py(self.log_name, net_name)
        save_yml(cfg, os.path.join('log/' + str(self.log_name), 'config.yml'))
        save_config(self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.train_dataset), len(self.train_loader)))
        save_config(self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.val_dataset), len(self.val_loader)))
        save_config(self.log_name, 'Model parameters: '+ str(sum(param.numel() for param in self.model.parameters())))

    def train(self): 
        scaler = GradScaler()
        with tqdm(total=len(self.train_loader), miniters=1,
                desc='Initial Training Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t:

            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                lr, hr = Variable(batch[0]), Variable(batch[1])
            #for data in self.train_loader:
                #lr, hr = data[0], data[1]
                if self.cuda:
                    lr, hr = lr.cuda(self.gpu_ids[0]), hr.cuda(self.gpu_ids[0])
                
                self.optimizer.zero_grad()               
                self.model.train()

                # if self.cfg['algorithm'] == 'SMSR':
                #     if self.cfg['use_apex']:
                #         with autocast():
                #             sr, sparsity = self.model(lr)
                #     else:
                #         sr, sparsity = self.model(lr)
                #     loss_sparsity = sparsity.mean()
                #     lambda0 = 0.1
                #     lambda_sparsity = min((self.epoch - 1) / 50, 1) * lambda0
                #     loss = self.loss(sr, hr) / (self.cfg['data']['batch_size'] * 2)
                #     epoch_loss += loss.data + lambda_sparsity * loss_sparsity
                #     del sparsity, lambda_sparsity
                # else:
                #     if self.cfg['use_apex']:
                #         with autocast():
                #             sr = self.model(lr)
                #     else:
                #         sr = self.model(lr)
                #     loss = self.loss(sr, hr) / (self.cfg['data']['batch_size'] * 2)
                #     epoch_loss += loss.data
                if self.cfg['schedule']['use_apex'] and self.cfg['gpu_mode']:
                        with autocast():
                            sr = self.model(lr)
                            loss = self.loss(sr, hr) / (self.cfg['data']['batch_size'] * 2)
                else:
                    sr = self.model(lr)
                    loss = self.loss(sr, hr) / (self.cfg['data']['batch_size'] * 2)
                epoch_loss += loss.data
                
                # if not self.cfg['schedule']['use_YCbCr']:
                #     loss = self.loss(sr, hr) / (self.cfg['data']['batch_size'] * 2)
                # else:
                #     import pytorch_colors as colors
                #     sr = colors.rgb_to_ycbcr(sr.detach()) / 255.0
                #     hr = colors.rgb_to_ycbcr(hr.detach()) / 255.0
                #     loss = self.loss(sr, hr) / (self.cfg['data']['batch_size'] * 2)

                epoch_loss += loss.data
                t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                t.update()

                if self.cfg['schedule']['use_apex'] and self.cfg['gpu_mode']:
                    scaler.scale(loss).backward()
                    # scaler.unscale_(self.optimizer)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # print("grad before clip:"+str(self.model.output_conv.conv.weight.grad))
                if self.cfg['schedule']['gclip'] > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg['schedule']['gclip']
                    )
                        
            self.records['Loss'].append(epoch_loss / len(self.train_loader))
            save_config(self.log_name, 'Initial Training Epoch {}: Loss={:.4f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)

    def eval(self):
        with tqdm(total=len(self.val_loader), miniters=1,
                desc='Val Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t1:
            psnr_list, ssim_list = [], []
            for iteration, batch in enumerate(self.val_loader, 1):
                lr, hr = Variable(batch[0]), Variable(batch[1])
            #for lr, hr in self.val_loader:
                if self.cuda:
                    lr, hr = lr.cuda(), hr.cuda()
                self.model.eval()
                with torch.no_grad():
                    if self.cfg['algorithm'] == 'SMSR':
                        sr, sparsity = self.model(lr)
                    else:
                        sr = self.model(lr)
                    loss = self.loss(sr, hr)

                batch_psnr, batch_ssim = [], []
                for c in range(sr.shape[0]):
                    if not self.cfg['data']['normalize']:
                        predict_sr = (sr[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        ground_truth = (hr[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                    else:          
                        predict_sr = (sr[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        ground_truth = (hr[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    psnr = calculate_psnr(predict_sr, ground_truth, 255)
                    ssim = calculate_ssim(predict_sr, ground_truth, 255)
                    batch_psnr.append(psnr)
                    batch_ssim.append(ssim)
                avg_psnr = np.array(batch_psnr).mean()
                avg_ssim = np.array(batch_ssim).mean()
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                t1.set_postfix_str('Batch loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}'.format(loss.item(), avg_psnr, avg_ssim))
                t1.update()
            self.records['Epoch'].append(self.epoch)
            self.records['PSNR'].append(np.array(psnr_list).mean())
            self.records['SSIM'].append(np.array(ssim_list).mean())

            save_config(self.log_name, 'Val Epoch {}: PSNR={:.4f}, SSIM={:.4f}'.format(self.epoch, self.records['PSNR'][-1],
                                                                    self.records['SSIM'][-1]))
            self.writer.add_scalar('PSNR_epoch', self.records['PSNR'][-1], self.epoch)
            self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], self.epoch)
            self.writer.add_image('image_SR', sr[0], self.epoch)
            self.writer.add_image('image_LR', lr[0], self.epoch)
            self.writer.add_image('image_HR', hr[0], self.epoch)

    def check_gpu(self):
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
            self.loss = self.loss.cuda(self.gpu_ids[0])

            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids) 

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['pretrain']['pre_sr'])
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['net'])
            self.epoch = torch.load(checkpoint, map_location=lambda storage, loc: storage)['epoch']
            for _ in range(self.epoch-1): self.scheduler.step()
            self.optimizer.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['optimizer'])
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self):
        super(Solver, self).save_checkpoint()
        self.ckp['net'] = self.model.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(self.cfg['checkpoint'] + '/' + str(self.log_name)):
            os.mkdir(self.cfg['checkpoint'] + '/' + str(self.log_name))
        torch.save(self.ckp, os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'))

        if self.cfg['save_best']:
            if self.records['PSNR'] != [] and self.records['PSNR'][-1] == np.array(self.records['PSNR']).max():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'best.pth'))

    def save_checkpoint_debug(self):
        super(Solver, self).save_checkpoint()
        self.ckp['net'] = self.model.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(self.cfg['checkpoint'] + '/' + str(self.log_name)):
            os.mkdir(self.cfg['checkpoint'] + '/' + str(self.log_name))
        torch.save(self.ckp, os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'))

    def run(self):
        self.check_gpu()
        if self.cfg['pretrain']['pretrained']:
            self.check_pretrained()
        try:
            while self.epoch <= self.nEpochs:
                self.train()
                if not self.cfg['debug']:
                    self.eval()
                    self.save_checkpoint()
                else:
                    self.save_checkpoint_debug()
                self.scheduler.step()
                self.epoch += 1
        except KeyboardInterrupt:
            self.save_checkpoint_debug()
        save_config(self.log_name, 'Training done.')