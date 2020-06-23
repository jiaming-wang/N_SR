#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 21:48:53
@LastEditTime: 2020-06-23 15:01:36
@Description: file content
'''

import re, yaml, os  

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

def save_yml(info, cfg_path):
    # if not os.path.exists(cfg_path):
    #     path_str = str(cfg_path)
    #     os.mkdir(os.path.join(cfg_path.split('/')[0], cfg_path.split('/')[1]))
    with open(cfg_path, 'w') as f:
        yaml.dump(info, f, Dumper=yaml.SafeDumper)

if __name__ == '__main__':
    config = get_config('./option.yml')
    #print(config)