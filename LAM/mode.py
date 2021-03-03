#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-19 07:02:16
LastEditTime: 2021-03-03 21:21:17
Description: file content
'''
import os, importlib
import torch


def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    # if model_name.split('-')[0] in NN_LIST:
    cfg = get_config('../option.yml')

    net_name = model_name.lower()
    lib = importlib.import_module('model.' + net_name)
    net = lib.Net
    net = net(
            num_channels=3, 
            base_filter=64,  
            scale_factor=4, 
            args = cfg).cuda()
    net = torch.nn.DataParallel(net, device_ids=[0,1])

    print_network(net, model_name)
    return net

def load_model(model_loading_name, checkpoint_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    net = get_model(model_loading_name)
    state_dict_path = os.path.join(checkpoint_name)
    print(f'Loading model {state_dict_path} for {model_loading_name} network.')
    state_dict = torch.load(state_dict_path, map_location=lambda storage, loc: storage.cuda(1))['net']
    net.load_state_dict(state_dict)
    return net

import re, yaml
def get_config(cfg_path):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
       u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.') 
    )
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=loader)
    return cfg
