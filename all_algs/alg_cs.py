#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/24 14:49
# @Author  : Lelsey
# @File    : alg_cs.py
import numpy as np

def alg_cs(y,A_rm,x_pc):
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
        links_state_infered=cs_diagnose(paths_state_obv,A_rm,x_pc)
        x_identified[:,i]=links_state_infered

    return x_identified

def cs_diagnose(paths_state_obv:np.ndarray,route_matrix:np.ndarray,links_cong_pro:list):
    """
    cs的核心代码
    :param paths_state_obv: array(m,)  路径的拥塞观测状态
    :param route_matrix:  array(m,n)  路由矩阵
    :return: link_state_inferred array(n,) 最终的推断链路
    """
    # 获取拥塞路劲的集合
    congested_paths,un_congested_paths=cal_cong_path_info(paths_state_obv)
    # print("拥塞路劲：", congested_paths)
    # print("不拥塞的路劲：", un_congested_paths)
    un_congested_links,suspected_congested_links=cal_cong_link_info(un_congested_paths,route_matrix)
    num_links=route_matrix.shape[1]
    links=set([i for i in range(num_links)])
    # print("所有链路：", links)
    # print("不拥塞的链路：", un_congested_links)
    # print("可以链路：", suspected_congested_links)

    QB = congested_paths
    # self.link_state_inferred = np.zeros(self.link_num)
    link_state_inferred = np.zeros(num_links)
    congested_link_inferred=[]

    while len(QB) != 0:
        # 从 E_C 中选择一条链路使得 gamma_k^(1) 取得最大值，并将此链路认为是拥塞链路
        # temp_state = [-1 for _ in range(self.link_num)]
        temp_state = [-1 for _ in range(num_links)]
        for i in suspected_congested_links:
            # temp_state[i] = np.log(1/self.link_congestion_Pr[i])
            temp_state[i] = np.log(1/links_cong_pro[i])
        link = temp_state.index(max(temp_state))
        link_state_inferred[link] = 1
        # self.congested_link_inferred.append(link + 1)
        congested_link_inferred.append(link + 1)

        suspected_congested_links = suspected_congested_links.difference(set([link]))

        domain = set()
        # for i in self.network.get_Path(link):
        for i in get_paths(link+1,route_matrix):
            if i not in un_congested_paths:
                domain.add(i)
        QB = QB.difference(domain)
    return link_state_inferred

def cal_cong_path_info(paths_state_obv):
    """
    根据路径的观测信息，计算拥塞路径和非拥塞路径
    :param paths_state_obv: array(m,)
    :return:paths_cong:set   paths_no_cong:set 存储路径下标
    """
    paths_cong = []
    paths_no_cong = []
    for index in range(len(paths_state_obv)):
        if int(paths_state_obv[index]) == 1:
            paths_cong.append(index )
        else:
            paths_no_cong.append(index )
    return set(paths_cong), set(paths_no_cong)

def cal_cong_link_info(un_congested_path:set,route_matrix:np.ndarray):
    """
    根据非拥塞路径，返回正常链路列表，和拥塞链路列表
    :param un_congested_path:list
    :return:good_link:set ,uncertain_link:set 存储链路下标
    """
    # 所有经过了不拥塞路径的链路
    good_link = []
    for i in un_congested_path:
        for index, item in enumerate(route_matrix[i]):
            if int(item) == 1 and index not in good_link:
                    good_link.append(index)

    num_links=route_matrix.shape[1]
    all_links = [i for i in range(num_links)]
    # 排除那些肯定不拥塞的链路
    uncertain_link = []
    for item in all_links:
        if item not in good_link:
            uncertain_link.append(item)
    return set(good_link), set(uncertain_link)

def get_paths(link: int,route_matrix):
    """
    获取经过指定链路的所有路径。

    在路由矩阵中，第 0 列代表链路 1，第 1 列代表链路 2。依次类推。
    第 0 行代表路径 1，第 1 行代表路径 2。依次类推。
    :param link: 链路的编号
    :return: paths  路径的下标：从0开始
    """
    assert link > 0
    paths, = np.where(route_matrix[:, link-1] > 0)
    return paths.tolist()

def test_cs():
    tree_vector = [0, 1, 1, 3, 3, 5, 5, 6, 6, 6]

    y = np.array([[1, 0, 1, 1, 1, 1],[1,1,1,1,1,1],[0,0,0,1,1,1]]).transpose()
    # y=np.array([[1,0,1,1,1,1],[1,1,1,1,1,1],[0,0,1,1,1,1]]).transpose()
    A_rm = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 1]])

    x_pc = np.array([0.1] * 10)

    links_state_infered = alg_cs(y, A_rm, x_pc)
    print(links_state_infered)

# def test2():
#     """
#     读取数据进行测试
#     :return:
#     """
#     import os
#     from all_DS.config import Config
#     from all_tools.utils import get_drfpr_f1j
# # file_path=os.path.join(os.path.dirname(os.getcwd()),'all_DS','TOPO_DS','[0, 1, 1, 3, 3].json')
#     file_path=os.path.join(os.path.dirname(os.getcwd()),'all_DS','TOPO_DS','[0, 1, 1, 3, 3, 5, 5, 6, 6, 6].json')
#     print(file_path)
#     conf=Config(file_path)
#     links_pro_scope=conf.get_links_pro_scope()
#     A_rm=conf.get_routing_matrix()
#     m, n = A_rm.shape  # 根据路由矩阵分别得到目标网络中路径与链路的数量
#
#     links_state_true=conf.get_links_state_true(str(links_pro_scope[0]))
#     route_matrix=conf.get_routing_matrix()
#     paths_state_obv=conf.get_paths_state_obv(str(links_pro_scope[0]))
#
#
#     links_cong_pro=conf.get_links_cong_pro(str(links_pro_scope[0]))
#     x_res=alg_cs(paths_state_obv,route_matrix,links_cong_pro)
#     print(x_res)
#
#     dr,fpr,f1,j=get_drfpr_f1j(links_state_true,x_res)
#     print("dr",dr)
#     print("fpr",fpr)
#     print("f1",f1)
#     print("j",j)

if __name__ == '__main__':
    # test_cs()
    # test2()