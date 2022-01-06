#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 14:54
# @Author  : Lelsey
# @File    : RM_to_vector.py

import numpy as np
from functools import cmp_to_key

def rm_to_vector(A_rm:np.ndarray):
    """
    将路由矩阵转换为树向量
    :param A_rm: np.ndarray(m,n)
    :return: vector (n,)
    """
    node_dist=[]
    a=A_rm.shape[1]
    tree_vector=[0]*(A_rm.shape[1])

    for i in range(A_rm.shape[0]):
        path=A_rm[i]
        pre_node=0
        for j in range(path.shape[0]):
            if path[j]==1:
                tree_vector[j]=pre_node
                pre_node=j+1

    return tree_vector

def rm_to_vector2(A_rm:np.ndarray):
    """
        将路由矩阵转换为树向量
        :param A_rm: np.ndarray(m,n)
        :return: vector (n,)
        """
    a = A_rm.shape[1]
    tree_vector = [0] * (A_rm.shape[1])

    for i in range(A_rm.shape[0]):
        path = A_rm[i]
        pre_node = 0
        for j in range(pre_node,path.shape[0]):
            if path[j] == 1:
                tree_vector[j] = pre_node
                pre_node+=1
    return tree_vector

def test1():
    rm_array = [[1, 1, 0, 0, 0], [1, 0, 1, 1, 0], [1, 0, 1, 0, 1]]
    A_rm = np.array(rm_array)
    tree_vector1 = rm_to_vector(A_rm)
    print(tree_vector1)
    tree_vector2 = rm_to_vector2(A_rm)
    print(tree_vector2)

def test2():
    pass


def gen_Tree_from_RM(routing_matrix):  # 依据路由矩阵来得到树型拓扑, 并没有检查路由矩阵的有效性
    """
    将路由矩阵转换为树向量
    :param routing_matrix: np.ndarray(m,n)
    :return: self.linkSetVec
    """
    routing_matrix = routing_matrix
    path_num, link_num = np.shape(routing_matrix)
    node_dist = np.arange(link_num + 1)
    tree_vector = [0] * (link_num)
    i_node = []
    for i in range(path_num):
        path = routing_matrix[i]
        pre_node = 0
        for j in range(path.size):
            if path[j] == 1:
                tree_vector[j] = pre_node
                i_node.append(pre_node)
                pre_node = j + 1

    dist = np.delete(node_dist, i_node)
    linkSetVec = np.asarray([tree_vector, range(1, link_num + 1)]).transpose()
    return linkSetVec

def rm_to_rm_sorted(A_rm:np.ndarray):
    """
    将路由矩阵转换为排序后的矩阵
    :param A_rm: np.ndarray(m,n)
    :return:  A_rm_sorted(m,n) 排序后的矩阵
    """
    y_list=column_sort(A_rm)
    A_rm_sorted=np.zeros(A_rm.shape)
    for i in range(len(y_list)):
        A_rm_sorted[:,i]=A_rm[:,y_list[i]]
    return A_rm_sorted

def column_cmp(y1,y2,A_rm):
    """
    列排序函数
    :param y1: y1列
    :param y2: y2列
    :param A_rm: np.ndarray(m,n)路由矩阵
    :return:
    """
    m=A_rm.shape[0]
    y1_set=set([i for i in range(m) if A_rm[i,y1]>0])
    y2_set=set([i for i in range(m) if A_rm[i,y2]>0])
    if y1_set.issubset(y2_set):
        return True
    return False


def column_sort(A_rm):
    """
    对矩阵的列进行排序
    :param A_rm: np.ndarray(m,n)
    :return: 正确的列编号
    """
    n=A_rm.shape[1]
    y_list=list(range(0,n))
    for i in range(n):
        for j in range(0,n-i-1):
            if column_cmp(y_list[j],y_list[j+1],A_rm):
                y_list[j],y_list[j+1]=y_list[j+1],y_list[j]
    print(y_list)
    return y_list

def test_column_sort():
    # A_rm_1=np.array([[1, 1, 0, 0, 0], [1, 0, 1, 1, 0], [1, 0, 1, 0, 1]])
    A_rm_2=np.array([[0,0,0,1,1],
                     [0,1,1,0,1],
                     [1,0,1,0,1]])
    A_rm_sorted=rm_to_rm_sorted(A_rm_2)
    print(A_rm_sorted)

if __name__ == '__main__':
    test_column_sort()
