#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 9:44
# @Author  : Lelsey
# @File    : pa_cs.py
#Network Tomography using Routing Probability for Virtualized Network中的算法实现

import cvxpy as cp
import numpy as np
import copy


def alg_pa_cs(y,A_rm,threshold=0.01):
    """
    综合的pa_cs算法，综合以下算法
    (1) pa_cs_1 (Lasso_Rgression) eq(3)  其他情况
    (2) pa_cs_2  eq(4)   非奇异矩阵
    (3) pa_cs_3  eq(5)   矩阵列满秩
    :param y: ndarray(m,)   ndarray(m,)路径的丢包率列向量 或者矩阵 array(m,s) 连续s次实验的丢包率矩阵
    :param A_rm: ndarray(m,n)   路由矩阵
    :param threshold float  丢包率门限
    :return: links_infer   链路的信息量
    """
    m, n = A_rm.shape
    _, num_times = y.shape

    #矩阵类型判断
    # flag==0 调用pa_cs_2
    # flag==1 调用pa_cs_3
    # falg==2 调用pa_cs_1
    flag=matrix_judge(A_rm)

    #数据预处理
    new_y=data_preprocessing(y)
    #算法开始

    x_identified = np.zeros((n, num_times))
    for i in range(num_times):
        paths_state_obv = y[:, i].flatten()
        if flag==2:
            links_state_infered = Lasso_Rgression(paths_state_obv, A_rm)
        elif flag==0:
            links_state_infered=pa_cs_2(paths_state_obv,A_rm)
        elif flag==1:
            links_state_infered=pa_cs_3(paths_state_obv,A_rm)

        x_identified[:, i] = links_state_infered

    new_x_identified=data_postprecessing(x_identified)

    return new_x_identified

def matrix_judge(A_rm):
    """
    判断矩阵是否为非奇异矩阵，列满秩矩阵，其他矩阵
    非奇异矩阵 返回0
    列满秩矩阵 返回1
    其他矩阵 返回2
    :param A_rm: array(m,n) 路由矩阵
    :return: flag  矩阵标志
    """
    m,n=A_rm.shape
    #矩阵的标志，默认普通矩阵
    flag=2

    #將矩阵补充为方阵
    Max=m if m>n else n
    A_rm_1=np.zeros(shape=(Max,Max))
    for i in range(A_rm.shape[0]):
        for j in range(A_rm.shape[1]):
            A_rm_1[i][j]=A_rm[i][j]

    A_rm_1_rank=np.linalg.matrix_rank(A_rm_1)
    if m==n and A_rm_1_rank==m:#非奇异矩阵
        flag=0
    elif A_rm_1_rank==n: #列满秩
        flag=1
    return flag

def data_preprocessing(y):
    """
    对数据的丢包率率进行预处理
    y into  -log(1-y)
    :param y: array(m,s) s次实验得到的路径丢包率
    :return: new_y:array(m,s)  数据的预处理结果
    """
    new_y=-1*np.log(1-y)
    return new_y

def data_postprecessing(x:np.ndarray):
    """
    对计算的结果进行还原处理
    x into 1-e^(-x)
    同时根据路径的丢包率计算推测的链路状态
    :param x: array(n,s) s次实验得到的链路丢包率
    :return: links_state_infer: array(n,s)  s次实验推测的链路状态
    """
    new_x=1-np.power(np.e,-1*x)

    links_state_infer=np.zeros(new_x.shape)
    for i in range(new_x.shape[0]):
        for j in range(new_x.shape[1]):
            if new_x[i,j]>=0.01:
                links_state_infer[i,j]=1

    return links_state_infer



#---------------------------eq(3)相关算法---------------
def Lasso_Rgression(y, A_rm, lambd_value=0.05):
    """
    pa_cs的算法1，
    计算公式：func=argmin_x{0.5* ||Ax-y||_2 + lambda*||x||_1 }
    :param y: ndarray(m,)   路径的丢包率列向量
    :param A_rm: ndarray(m,n)   路由矩阵
    :param lambd_value: float   lambda参数
    :return: links_state_infer:naarray(n,)   链路的丢包率列向量
    """

    m, n = A_rm.shape

    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)
    lambd.value = lambd_value
    problem = cp.Problem(cp.Minimize(objective_fn(A_rm, y, beta, lambd)))
    problem.solve()

    links_state_infer = beta.value

    return links_state_infer

def loss_fn(X, Y, beta):
    #第二范式
    return cp.norm2(X @ beta - Y)**2

def regularizer(beta):
    #第一范式
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    #func=argmin_x{0.5* ||Ax-y||_2 + lambda*||x||_1 }
    #目标方程
    return 0.5*loss_fn(X, Y, beta) + lambd * regularizer(beta)


#----------------------------eq(4)算法
def pa_cs_2(y,A_rm):
    """
    实现公式 x=A^-1 *y
    :param y: array(m,1)  路径的丢包率数组
    :param A_rm: array(m,n) 路由矩阵
    :return: links_loss_infer array(n,)  推断的链路丢包率数组
    """
    A_rm_1=np.linalg.inv(A_rm)
    x=np.dot(A_rm_1,y)
    return x.flatten()

#----------------------------eq(5)算法
def pa_cs_3(y,A_rm):
    """
    实现公式x=(A^T * A)^-1 *A^T *y
    :param y: array(m,1)  路径的丢包率数组
    :param A_rm: array(m,n) 路由矩阵
    :return: links_loss_infer array(n,)  推断的链路丢包率数组
    """
    A_t_A_1=np.linalg.inv(A_rm.T @ A_rm)
    x=A_t_A_1 @ A_rm.T @ y
    return x.flatten()

#-------------------------------以下是测试文件------------------

def test_pacs_1():

    y=np.array([0.0010733008099377273,
                0.016900289954767667,
                0.017810402199125663])

    print("预处理前的y")
    print(y)
    y_1=-1*np.log(1-y)
    print("预处理后的y")
    print(y_1)

    A_rm = np.array([[1,1,0,0,0],
                     [1,0,1,1,0],
                     [1,0,1,0,1]])

    links_state_infer=Lasso_Rgression(y_1,A_rm)

    links_state_infer=data_postprecessing(links_state_infer)
    print(links_state_infer)

def test_alg_pa_cs():
    #d对pa_cs算法进行简单测试
    y = np.array([[0.0010733008099377273, 0.016900289954767667, 0.017810402199125663],
                  [0.007823569609156755, 0.020921555155804517, 0.019866611188801175]]).transpose()

    A_rm = np.array([[1, 1, 0, 0, 0],
                     [1, 0, 1, 1, 0],
                     [1, 0, 1, 0, 1]])
    links_state_infer=alg_pa_cs(y,A_rm)
    print(links_state_infer)


#进行大规模的测试
# from all_DS.config import Config
# import os
# from all_tools import utils
# def test_large_scale():
#     # file_path=os.path.join(os.path.dirname(os.getcwd()),'all_DS','TOPO_DS','[0, 1, 1, 3, 3].json')
#     file_path=os.path.join(os.path.dirname(os.getcwd()),'all_DS','TOPO_DS','[0, 1, 1, 3, 3, 5, 5, 6, 6, 6].json')
#     print(file_path)
#     conf=Config(file_path)
#     links_pro_scope=conf.get_links_pro_scope()
#     paths_loss_rate=conf.get_paths_loss_rate(str(links_pro_scope[0]))
#     links_state_true=conf.get_links_state_true(str(links_pro_scope[0]))
#     route_matrix=conf.get_routing_matrix()
#
#     x_res=alg_pa_cs(paths_loss_rate,route_matrix)
#     print(x_res)
#
#     dr,fpr,f1,j=utils.get_drfpr_f1j(links_state_true,x_res)
#     print("dr",dr)
#     print("fpr",fpr)
#     print("f1",f1)
#     print("j",j)


    pass

if __name__ == '__main__':
    # test_pacs_1()
    # test_alg_pa_cs()
    # test_large_scale()
    pass