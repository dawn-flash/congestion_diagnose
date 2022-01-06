#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/2 14:13
# @Author  : Lelsey
# @File    : alg_map_advace.py

"""
map的改进算法。
求出所有不确定链路的最佳组合使得，在符合路径拥塞的情况下使得该组合的概率最大
策略利用dfs递归+剪枝删除不需要遍历的决策
"""
import numpy as np
import copy


def alg_map_advace(y: np.ndarray, A_rm: np.ndarray, x_pc: np.ndarray):
    """
    clink算法的调用接口
    :param y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
    :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
    :param x_pc: ’probability of congestion' 列向量
    :return: x_identified 为返回的已识别的链路拥塞状态；列向量，如为矩阵，横维度为时间，纵维度为链路推测状态
    """

    if np.ndim(y) <= 1:  # 强制转换为列向量
        y = y.reshape((-1, 1))

    m, n = A_rm.shape
    _, num_time = y.shape

    x_identified = np.zeros((n, num_time),dtype=int)
    for i in range(num_time):
        paths_state_obv = y[:, i]
        links_state_infered, res_prob= map_a_group(paths_state_obv, A_rm, x_pc)
        x_identified[:, i] = links_state_infered
    return x_identified

def map_a_group(y,A_rm,x_pc):
    """
    map_剪枝算法 算法测试一组数据
    :param y: np.ndarray(path,)  所有观测路径的拥塞状态，普通向量
    :param A_rm: np。ndarray(path,link)  路由矩阵
    :param x_pc: 链路拥塞概率，np.ndarray(link,)  普通向量
    :return:links_state_inferred: np.ndarray(link,) 最终链路推断的结果  true_pro  推断结果对应的概率
    """
    net_attri = {
        "link_pro": x_pc.tolist(),
        "uncer_link_number": [],
        "route_matrix": A_rm.tolist(),
        "congest_path": y.tolist()
    }

    #不确定链路的编号
    uncer_link_number=cal_uncer_link_num(net_attri["congest_path"],net_attri["route_matrix"])
    net_attri["uncer_link_number"]=uncer_link_number

    link_num=len(A_rm[0])
    links_state_inferred=[0]*link_num

    max_depth = len(uncer_link_number)
    depth = -1
    cur_pro = 1
    best_pro = [0]
    best_scene = [0] * (max_depth)
    uncer_link = copy.deepcopy(best_scene)

    #需要判断时进行dfs算法判断
    if max_depth!=0:
        dfs(uncer_link,net_attri,depth,max_depth,cur_pro,best_pro,best_scene)

        #根据不确定场景恢复所有的推测链路状态 和实际概率
        for i in range(len(best_scene)):
            if best_scene[i]==1:
                num=uncer_link_number[i]
                links_state_inferred[num-1]=1

    # print(uncer_link_number)

    #计算链路推测结果的概率
    true_pro = 1
    for i in range(len(links_state_inferred)):
        if links_state_inferred[i] == 1:
            true_pro *= x_pc[i]
        if links_state_inferred[i] == 0:
            true_pro *= (1 - x_pc[i])
    return np.array(links_state_inferred),true_pro



def cal_uncer_link_num(congest_path,route_matrix):
    """
    求在拥塞路径状态下，不确定链路的编号列表
    :param congest_path: list(path,)  拥塞路径状态
    :param route_matrix:  list(path,link)  路由矩阵
    :return:  uncer_link_number ： list(n,) n个不确定链路的编号，编号从1开始计算
    """
    link_num=len(route_matrix[0])
    link_state=[1]*link_num
    for i in range(len(congest_path)):
        if congest_path[i]==0:
            for j in range(link_num):
                if route_matrix[i][j]==1:
                    link_state[j]=0

    uncer_link_number=[i+1 for i in range(len(link_state)) if link_state[i]==1]
    return uncer_link_number




def dfs(uncer_link,net_attri,depth,max_depth,cur_pro,best_pro,best_scene):
    """
    net_attri{
        link_pro:list(link,) 所有链路的拥塞概率
        uncer_link_number:list(n,)     不确定链路对应的编号[1,2,3..] 链路的编号从1开始
        route_matrix:list(path,link) 路由矩阵
        congest_path:list(path,) 所有拥塞路径列表 01列表
    }
    :param uncer_link: list(n,) 当前n条不确定链路
    :param net_attri: dict 网络的相关属性
    :param depth: number 当前决策遍历的深度   深度和节点对应， 深度0 对应根节点，无效
    :param max_depth: number  最大深度
    :param cur_pro: number 当前结点的概率
    :param best_pro: list(1,) 最佳组合的概率  用于保存最佳概率
    :param best_scene: list（1,link）link条链路最佳组合对应的场景
    :return:
    """

    # print()
    # print("当前遍历  深度{}  链路{}   最好概率{}  最好组合{}--计算前".format(depth, uncer_link, best_pro, best_scene))

    if depth!=-1:

        #判断链路的拥塞场景是否符合路径的拥塞场景
        flag=judge_scene(uncer_link,depth,net_attri)
        if not flag:
            # print()
            # print("路径剪枝  深度{}  链路{}   最好概率{}  最好组合{}".format(depth, uncer_link, best_pro, best_scene))
            return

        #计算当前结点拥塞的概率
        link_prob=net_attri["link_pro"][net_attri["uncer_link_number"][depth]-1]
        cur_pro=cur_pro*(link_prob*uncer_link[depth]+(1-link_prob)*(1-uncer_link[depth]))
        if depth==max_depth-1:
            if best_pro[0]<cur_pro:
                best_pro[0]=cur_pro
                for i in range(len(uncer_link)):
                    best_scene[i]=uncer_link[i]
            return
        if cur_pro<best_pro[0]:
            # print()
            # print("概率剪枝  深度{}  链路{}   最好概率{}  最好组合{}".format(depth,uncer_link,best_pro,best_scene))
            return

    # print()
    # print("当前遍历  深度{}  链路{}   最好概率{}  最好组合{}--计算后".format(depth, uncer_link, best_pro, best_scene))

    #当前节点的决策为0
    uncer_link[depth+1]=0
    dfs(uncer_link,net_attri,depth+1,max_depth,cur_pro,best_pro,best_scene)
    uncer_link[depth + 1] = 0

    #当前节点的决策为1
    uncer_link[depth+1]=1
    dfs(uncer_link,net_attri,depth+1,max_depth,cur_pro,best_pro,best_scene)
    uncer_link[depth+1]=0



def judge_scene(uncer_link,depth,net_attri):
    """
    判断链路的钥匙呢场景是否符合路径的拥塞场景
    net_attri{
        link_pro:list(link,) 所有链路的拥塞概率
        uncer_link_number:list(link,)不确定链路对应的编号[1,2,3]
        route_matrix:list(path,link) 路由矩阵
        congest_path:list(path,) 所有拥塞路径列表 01列表
    }
    :param uncer_link: list(link,) 条不确定链路,
    :param depth number : uncer_link
    :param net_attri: 网络的相关属性
    :return: flag   符合条件返回true，不符合情况返回false
    """
    link_num=len(net_attri["link_pro"])
    path_num=len(net_attri["congest_path"])

    #构建当前链路状态 确定状态为0，不确定状态为1
    link_scene=[0]*link_num
    for l in net_attri["uncer_link_number"]:
        link_scene[l-1]=1

    #根据确定的depth条链路，改变已经确认的链路
    for i in range(depth+1):
        if(uncer_link[i]==0):
            num=net_attri["uncer_link_number"][i]
            link_scene[num-1]=0

    link_scene=np.array(link_scene,dtype=int).reshape((link_num,1))
    route_matrix=np.array(net_attri["route_matrix"],dtype=int)
    res_path=np.dot(route_matrix,link_scene)

    #判断是否符合条件 如果出现，推测为0，原来为1的情况，直接返回false
    flag=True
    for i in range(len(net_attri["congest_path"])):
        if net_attri["congest_path"][i]==1 and res_path[i][0]==0:
            flag=False
            break

    return flag

#------------------------------------------以下是测试代码
def test_judge_scene():
    """
    测试judge_scene函数的有效性
    :return:
    """
    # uncer_link=[0,0,1,1,1]  #false
    uncer_link=[0,0,0,0,0]  #false
    depth=1
    net_attri={
        "link_pro": [0.1,0.1,0.1,0.1,0.1],
        "uncer_link_number":[1,2,3,4,5] ,
        "route_matrix": [[1,1,0,0,0],
                         [1,0,1,1,0],
                         [1,0,1,0,1]],
        "congest_path": [1,1,1]
    }
    flag=judge_scene(uncer_link,depth,net_attri)
    print(flag)

def test_dfs():

    # #案例1
    # net_attri = {
    #     "link_pro": [0.1,0.8,0.9,0.1,0.1],
    #     "uncer_link_number": [3, 4, 5],
    #     "route_matrix": [[1, 1, 0, 0, 0],
    #                      [1, 0, 1, 1, 0],
    #                      [1, 0, 1, 0, 1]],
    #     "congest_path": [1, 1, 1]
    # }

    # # 案例2
    # net_attri = {
    #     "link_pro": [0.1, 0.8, 0.1, 0.9, 0.9],
    #     "uncer_link_number": [3, 4, 5],
    #     "route_matrix": [[1, 1, 0, 0, 0],
    #                      [1, 0, 1, 1, 0],
    #                      [1, 0, 1, 0, 1]],
    #     "congest_path": [0, 1, 1]
    # }

    # # 案例3
    # net_attri = {
    #     "link_pro": [0.1, 0.1, 0.1, 0.9, 0.4],
    #     "uncer_link_number": [2,4],
    #     "route_matrix": [[1, 1, 0, 0, 0],
    #                      [1, 0, 1, 1, 0],
    #                      [1, 0, 1, 0, 1]],
    #     "congest_path": [1,1,0]
    # }

    # 案例4
    net_attri = {
        "link_pro": [0.1, 0.5, 0.1, 0.9, 0.4],
        "uncer_link_number": [2],
        "route_matrix": [[1, 1, 0, 0, 0],
                         [1, 0, 1, 1, 0],
                         [1, 0, 1, 0, 1]],
        "congest_path": [1, 0, 0]
    }



    max_depth=len(net_attri["uncer_link_number"])
    depth=-1
    cur_pro=1
    best_pro=[0]
    best_scene=[0]*(max_depth)

    uncer_link=copy.deepcopy(best_scene)

    dfs(uncer_link,net_attri,depth,max_depth,cur_pro,best_pro,best_scene)
    print("best_prob",best_pro)
    print("best_scene",best_scene)


def test_cal_uncer_link_num():
    congest_path=[1,0,0]
    route_matrix=[[1, 1, 0, 0, 0],
                         [1, 0, 1, 1, 0],
                         [1, 0, 1, 0, 1]]
    uncer_link_number=cal_uncer_link_num(congest_path,route_matrix)
    print(uncer_link_number)

def test_map_a_group():
    """
    测试map算法
    :return:
    """

    #测试案例1
    y=np.array([1, 1, 1])
    A_rm=np.array([[1, 1, 0, 0, 0],
                         [1, 0, 1, 1, 0],
                         [1, 0, 1, 0, 1]])
    x_pc=np.array([0.1,0.8,0.9,0.1,0.1])

    #测试案例2
    y = np.array([0 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1])
    A_rm = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
 [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
 [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
    x_pc = np.array([0.549, 0.715, 0.603, 0.545, 0.424, 0.646, 0.438, 0.892, 0.964, 0.383, 0.792, 0.529, 0.568, 0.926, 0.071, 0.087,
     0.02, 0.833, 0.778])



    links_state_inferred,best_pro=map_a_group(y,A_rm,x_pc)
    print(links_state_inferred)
    print(best_pro)

    res=1
    for i in range(len(links_state_inferred)):
        if links_state_inferred[i]==1:
            res*=x_pc[i]
        if links_state_inferred[i]==0:
            res*=(1-x_pc[i])
    print("res",res)


def test_alg_map_advace():
    """
    测试alg_map_advace 算法接口
    :return:
    """
    # 测试案例1
    y = np.array([1, 1, 1])
    A_rm = np.array([[1, 1, 0, 0, 0],
                     [1, 0, 1, 1, 0],
                     [1, 0, 1, 0, 1]])
    x_pc = np.array([0.1, 0.8, 0.9, 0.1, 0.1])

    # 测试案例2
    y = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    A_rm = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
    x_pc = np.array(
        [0.549, 0.715, 0.603, 0.545, 0.424, 0.646, 0.438, 0.892, 0.964, 0.383, 0.792, 0.529, 0.568, 0.926, 0.071, 0.087,
         0.02, 0.833, 0.778])

    links_state_inferred = alg_map_advace(y, A_rm, x_pc)

    print(links_state_inferred)








def test():
    # l=np.array([[1,1],[2,2],[3,3]])
    # h=np.array([[1],[1]])
    # res=np.dot(l,h)
    # print(res)
    if True:
        l=1
    print(l)


if __name__ == '__main__':
   # test()
   # test_judge_scene()
   # test_dfs()
   #  test_cal_uncer_link_num()
   #  test_map_a_group()
   test_alg_map_advace()