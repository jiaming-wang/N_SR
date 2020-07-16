#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-07-16 15:10:34
@LastEditTime: 2020-07-16 15:31:28
@Description: file content
'''
class People(object):
    def __init__(self,lable1):

        self.lable1= lable1

    def fun(self):
        print("fun")

#子类
class Student(People):
    def __init__(self,score):
        self.score = score

    #重写;将函数的声明和实现重新写一遍
    def fun(self):
        #在子类函数中调用父类中的函数【1.想使用父类中的功能，2.需要添加新的功能】
        #根据具体的需求决定需不需要调用父类中的函数
        super(Student,self).fun()
        print("fajfhak")

class Student1(Student):
    def __init__(self,name,score, lable1, lable2):
        self.lable2 = lable2
        super().__init__(score)

    #重写;将函数的声明和实现重新写一遍
    def fun(self):
        print(self.score)
        
s = Student1(1,2,3,4)
s.fun()