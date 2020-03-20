#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2019-10-13 23:32:02
@Description: main.py
'''

from utils.config import get_config
from solver.solver import Solver

if __name__ == '__main__':
    cfg = get_config('option.yml')
    solver = Solver(cfg)
    solver.run()
    