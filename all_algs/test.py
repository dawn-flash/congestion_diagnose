#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 15:42
# @Author  : Lelsey
# @File    : test.py
import numpy as np
import matplotlib.pyplot as plt

def test1(a):
    print("函数内")
    print(a)
    a[0]=2
    print(a)
    print("函数内")

def test2():
    #计算log
    # a=np.array([[0.0010733,0.00782357], [0.01690029,0.02092156], [0.0178104 ,0.01986661]])
    # # print(1-a)
    # print(-np.log(1-a[0,0]))
    # b=-np.log(1-a)
    # print(b)

    #计算次方
    x=np.array([[9.84473059e-10,8.14973791e-10], [3.85745871e-11,4.86571373e-12],
        [6.92350594e-10,7.18005952e-11], [1.84198583e-10,1.48533637e-11],
        [1.90454179e-10,1.38447796e-11]])
    new_x=1-np.power(np.e,-1*x)
    print(new_x)

def test_3():
    a=np.array([[np.e,2],[3,4]])
    # b=np.array([1,1]).reshape((-1,1))
    b=np.array([1,1]).reshape((-1,1))
    a[:,0]=b

    print(a)

    #矩阵的逆
    # d=np.array([[1,0,-2],[-3,1,4],[2,-3,4]])
    # e=np.linalg.inv(d)
    # print(e)


    #矩阵的转置
    # d=np.array([[1,0,-2],[-3,1,4],[2,-3,4]])
    # print(d.T)

    #矩阵行列式
    # a=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]])
    # a_rank=np.linalg.matrix_rank(a)
    # print(a_rank)
    # b=np.linalg.det(a)
    # print(b)

    #计算矩阵的行秩
    # a=np.array([[1,0],[0,1],[1,1]])
    # m,n=a.shape
    # Max=m if m>n else n
    # a_1=np.zeros(shape=(Max,Max))
    # for i in range(a.shape[0]):
    #     for j in range(a.shape[1]):
    #         a_1[i][j]=a[i][j]
    # print(a_1)
    # a_rank=np.linalg.matrix_rank(a_1)
    # print(a_rank)

def test4():
    # x_lossrate_estimated=np.array([[1,2,3,4,5],[1,2,3,4,5]])
    # x_identified = (x_lossrate_estimated > 3).astype(np.int8)
    #
    # print(x_identified)
    print(np.log(0))
def test5():
    a=set([1,2,3])
    for i in a:
        print(i)


if __name__ == '__main__':
    # test2()
    # test_3()
    # test4()
    test5()