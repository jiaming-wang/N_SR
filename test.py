#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-02-18 14:00:21
@Description: test.py
'''

from utils.config import get_config
from solver.testsolver import Testsolver

if __name__ == '__main__':
    cfg = get_config('option.yml')
    solver = Testsolver(cfg)
    solver.run()