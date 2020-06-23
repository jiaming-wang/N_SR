#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 14:47:44
@Description: test.py
'''

from utils import get_config
from solver import Testsolver

if __name__ == '__main__':
    cfg = get_config('option.yml')
    solver = Testsolver(cfg)
    solver.run()
    