#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 14:31
# @Author  : Lelsey
# @File    : clink_0.py

import numpy as np
import random
from all_algs.clink import CLINK
from all_nets.cong_net import Cong_net
from all_tools import utils
import copy

def alg_clink(y,A_rm,x_pc):
    """
    clink算法的调用接口
    :param y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
    :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
    :param x_pc: ’probability of congestion' 列向量
    :return: x_identified 为返回的已识别的链路拥塞状态；列向量，如为矩阵，横维度为时间，纵维度为链路推测状态
    """
    m,n=A_rm.shape
    _,num_times=y.shape

    x_identified=np.zeros((n,num_times))
    for i in range(num_times):
        paths_state_obv=y[:,i]
        links_state_infered=clink_a_groub(paths_state_obv,A_rm,x_pc)
        x_identified[:,i]=links_state_infered

    return x_identified


def clink_a_groub(y,A_rm,x_pc):
    """
    clink 算法测试一组数据
    :param y: np.ndarray  观测路径的拥塞状态，普通向量
    :param A_rm: np。ndarray  路由矩阵
    :param x_pc: 链路拥塞概率，np.ndarray  普通向量
    :return:links_state_inferred: np.ndarray 普通
    """
    tree_vector = rm_to_vector(A_rm)
    links_congest_pro = list(x_pc.flatten())
    net = Cong_net()
    net.auto_init(tree_vector, "loss_model_1", 0.015, links_congest_pro)

    clink=CLINK(net)
    clink.paths_state_obv = copy.deepcopy(y)  # 观测路径状态数组
    clink.paths_cong_obv ,clink.paths_no_cong_obv =cal_cong_path_info(y)
    clink.diagnose()
    return clink.links_state_inferred

def cal_cong_path_info(paths_state_obv):
    """
    根据路径的观测信息，计算拥塞路径和非拥塞路径
    :param paths_state_obv:
    :return:
    """
    paths_cong = []
    paths_no_cong = []
    for index in range(len(paths_state_obv)):
        if int(paths_state_obv[index]) == 1:
            # if int(self.path_states[index]) == 1:
            paths_cong.append(index + 1)
        else:
            paths_no_cong.append(index + 1)
    return np.array(paths_cong), np.array(paths_no_cong)

def rm_to_vector(A_rm:np.ndarray):
    """
    将路由矩阵转换为树向量
    :param A_rm:
    :return:
    """
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

def test_clink_alg():
    #测试一
    # tree_vector=[0, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 9, 9, 12, 12, 15, 15]
    # net = Cong_net()
    # net.auto_init(tree_vector, "loss_model_1", 0.015, 0.1)
    # y=np.array([0 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1]).reshape((-1,1))
    # A_rm=net.route_matrix
    # x_pc=np.array([0.549, 0.715, 0.603, 0.545, 0.424, 0.646, 0.438, 0.892, 0.964, 0.383, 0.792, 0.529, 0.568, 0.926,
    #                   0.071, 0.087, 0.02, 0.833, 0.778])



    #测试二
    tree_vector = [0, 1, 1, 3, 3, 5, 5, 6, 6, 6]

    net = Cong_net()
    net.auto_init(tree_vector, "loss_model_1", 0.015, 0.1)
    y=np.array([1,0,1,1,1,1]).reshape((-1,1))
    # y=np.array([[1,0,1,1,1,1],[1,1,1,1,1,1],[0,0,1,1,1,1]]).transpose()
    A_rm=net.route_matrix
    x_pc=np.array([0.1]*10)

    links_state_infered = alg_clink(y, A_rm, x_pc)
    print(links_state_infered)


if __name__ == '__main__':
    test_clink_alg()