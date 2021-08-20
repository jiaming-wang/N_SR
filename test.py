#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
LastEditTime: 2021-08-20 23:41:00
@Description: test.py
'''
import argparse
from utils.config  import get_config
from solver.testsolver import Testsolver

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N_SR_test')
    parser.add_argument('--option_path', type=str, default='option.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    solver = Testsolver(cfg)
    solver.run()
    