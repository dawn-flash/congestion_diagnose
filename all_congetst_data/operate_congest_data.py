#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 16:31
# @Author  : Lelsey
# @File    : gen_congest_data.py

"""
操作all_topo文件中的数据，生成每一个网络拓扑的测量数据保存到data目录下
"""

import numpy as np
import os
import json
from all_tools import utils
from pprint import pprint
import copy


def gen_congest_isomorphism(tree_vector, route_matrix, time, prob):
    """
    在同构链路失效概率下，生成链路拥塞矩阵（链路，次数），路径拥塞矩阵（路径，次数）
    :param tree_vector: list 树向量
    :param route_matrix: list 路由矩阵(path,link)
    :param time:  number 测量次数
    :param prob:  number 概率
    :return: link_congest_M:list 维度(link.time),path_congest_M:list 维度(path,time)
    """
    link_number = len(tree_vector)
    prob_list = [prob for i in range(link_number)]
    link_congest_M = np.random.uniform(0, 1, (link_number, time))

    # print("输出概率矩阵")
    # pprint(link_congest_M)

    for i in range(link_congest_M.shape[0]):
        for j in range(link_congest_M.shape[1]):
            if link_congest_M[i, j] <= prob:
                link_congest_M[i, j] = 1
            else:
                link_congest_M[i, j] = 0

    route_matrix = np.array(route_matrix)
    path_congest_M = np.dot(route_matrix, link_congest_M)
    path_congest_M = np.where(path_congest_M < 1, path_congest_M, 1)

    # 将array转为整形，然后转为list
    link_congest_M=link_congest_M.astype(int)
    link_congest_M = link_congest_M.tolist()
    path_congest_M=path_congest_M.astype(int)
    path_congest_M = path_congest_M.tolist()

    return link_congest_M, path_congest_M


def gen_congest_isomerism(tree_vector, route_matrix, time, prob_scope, K):
    """

    :param tree_vector: list 树向量
    :param route_matrix: list 路由矩阵(path,link)
    :param time:  number 测量次数
    :param prob_scope: 概率区间 list （2,）
    :param K:  每一个概率区间，生成k组概率组
    :return:  link_congest_M:list 维度(k,link,time)  ,path_congest_M:list 维度(k,path,time)  link_prob_list(k,link) k个链路组
    """
    link_congest_MK = []
    path_congest_MK = []
    link_prob_list = []
    link_num = len(tree_vector)

    for k in range(K):
        # 生成链路概率组
        prob_list = np.random.uniform(prob_scope[0], prob_scope[1], link_num)
        # 生成一组测量矩阵
        link_congest_M = np.random.uniform(0, 1, (link_num, time))

        # 测量矩阵转换为[0,1]矩阵  根据链路概率
        for i in range(link_congest_M.shape[0]):
            for j in range(link_congest_M.shape[1]):
                if link_congest_M[i, j] <= prob_list[i]:
                    link_congest_M[i, j] = 1
                else:
                    link_congest_M[i, j] = 0

        # 根据链路测量矩阵和路由矩阵 得到测量路径矩阵
        route_matrix = np.array(route_matrix)
        path_congest_M = np.dot(route_matrix, link_congest_M)
        path_congest_M = np.where(path_congest_M < 1, path_congest_M, 1)

        # 将array转为list  矩阵中的元素转为int类型
        link_congest_M=link_congest_M.astype(int)
        link_congest_M = link_congest_M.tolist()
        path_congest_M=path_congest_M.astype(int)
        path_congest_M = path_congest_M.tolist()
        prob_list = prob_list.tolist()

        # 添加到k次的列表中
        link_congest_MK.append(link_congest_M)
        path_congest_MK.append(path_congest_M)
        link_prob_list.append(prob_list)

        # print("链路结果")
        # print(link_congest_M)
        # print("路径结果")
        # print(path_congest_M)
        # print("链路概率")
        # print(prob_list)

    # print("k次链路处理完毕")



    return link_congest_MK, path_congest_MK, link_prob_list


def gen_a_congest_data(tree_vector, tree_name, measure_time, K, link_prob_isomorphism, link_prob_isomerism):
    """
    生成一个拓扑的拥塞数据
    :param tree_vector: list 树向量
    :param tree_name: str 拓扑名字
    :param route_matrix:  list 路由矩阵 (path,link)\
    :param measure_time: number 测量次数
    :param K: number  K组异构概率组
    :return:
    """
    route_matrix = utils.tree_vector_to_route_matrix(tree_vector).tolist()
    congest_data = {
        "tree_name": tree_name,
        "tree_vector": tree_vector,
        "route_matrix": route_matrix,
        "measure_time": measure_time,
        "link_prob_isomorphism":link_prob_isomorphism,
        "link_prob_isomerism":link_prob_isomerism,
        "K":K,
        "isomorphism": {},
        "isomerism": {}
    }

    # 生成同构概率链路的数据
    link_number = len(tree_vector)
    for i in range(len(link_prob_isomorphism)):
        prob = link_prob_isomorphism[i]
        link_prob_list = [prob] * link_number
        link_measure_data, path_measure_data = gen_congest_isomorphism(tree_vector, route_matrix, measure_time, prob)
        a_group_data = {
            "link_prob_list": link_prob_list,
            "link_measure_data": link_measure_data,
            "path_measure_data": path_measure_data
        }
        congest_data["isomorphism"][str(i)] = a_group_data

    # 生成异构概率链路的数据
    for i in range(len(link_prob_isomerism)):
        prob_scope = link_prob_isomerism[i]
        link_measure_data_k, path_measure_data_k, link_prob_list = gen_congest_isomerism(tree_vector, route_matrix,
                                                                                         measure_time,
                                                                                         prob_scope, K)
        a_group_data_k = {
            "prob_scope": prob_scope,
            "link_prob_list": link_prob_list,
            "link_measure_data": link_measure_data_k,
            "path_measure_data": path_measure_data_k
        }
        congest_data["isomerism"][str(i)] = a_group_data_k

    return congest_data


def gen_all_congest_data():
    """
    从"all_topo.json"中读取数据
    生成所有的真实的测量数据 数据保存在data文件中
    json格式
    congest_data = {
        "tree_name": tree_name,
        "tree_vector": tree_vector,
        "route_matrix": route_matrix,
        "measure_time": measure_time,
        "isomorphism": {},
        "isomerism": {}
    }
    K：异构链路中，一组概率范围选出的概率组数量 目前取20
    :return:
    """
    # 读取json数据
    file_read_path = "all_topo.json"
    all_topo_data = {}
    with open(file_read_path, 'r') as fr:
        all_topo_data = json.load(fr)

    measure_time = 1000
    K = 20
    link_prob_isomorphism = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    link_prob_isomerism = [[i * 0.5, i * 1.5] for i in link_prob_isomorphism]
    dir_name = os.path.join(os.getcwd(), "data_true")

    cur_i=0
    for kv in all_topo_data.items():
        topo = kv[1]
        tree_name = topo["name"]
        tree_vector = topo["tree_vector"]
        congest_data = gen_a_congest_data(tree_vector, tree_name, measure_time, K, link_prob_isomorphism,
                                          link_prob_isomerism)
        print("-------------当前处理第"+str(cur_i)+"个文件-----------------")
        save_json(tree_name, congest_data, dir_name)
        cur_i=cur_i+1


def save_json(tree_name, congest_data, dir_name):
    """
    保存真实测量数据到以dir目录下
    :param tree_vector: list 树向量
    :param tree_name:   str  树名字
    :param congest_data:  dict 拥塞数据
    :param dir_name:  str   目录名字
    :return:
    """
    # 保存数据
    file_path = os.path.join(dir_name, tree_name + '.json')
    print("保存文件开始", file_path)
    with open(file_path, 'w')as fw:
        json.dump(congest_data, fw, indent=4)
    print("保存文件结束", file_path)


#---------------------生成带有误差的路径观测数据--------------------

def gen_all_path_obv_data():
    """
    从data_true文件夹中读取所有的的json数据
    生成带有一定误差的观测数据
    保存到data_obv中
    :return:
    """

    #获取data文件夹下的所有文件
    source_dir=os.path.join(os.getcwd(),'data_true')
    target_dir=os.path.join(os.getcwd(),'data_obv')
    file_list=os.listdir(source_dir)

    error_list=[0.0,0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]

    #循环生成每一个topo的文件
    for i in range(len(file_list)):
        source_file_path=os.path.join(source_dir,file_list[i])
        target_file_path=os.path.join(target_dir,file_list[i])
        print("----------------------开始处理第"+str(i)+"个拓扑----------------")
        gen_a_path_obv_data(source_file_path,target_file_path,error_list)

    print("程序运行完毕")



def gen_a_path_obv_data(source_path,target_path,error_list):
    """
    根据data文件下所有topo的真实链路和路径的拥塞数据 按照
    [0.0,0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25] 的误差率生成仿真实的观测数据
    数据保存到obv_data文件中
    :param source_path: str 真实拥塞路径数据   源
    :param target_path: str  观测的拥塞路径数据  目标
    :param error_list: list(n,)
    :return:
    """

    #从source_path中读取数据
    true_congest_data={}
    with open(source_path,'r') as fr:
        true_congest_data=json.load(fr)

    # link_prob_isomorphism = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    # link_prob_isomerism = [[i * 0.5, i * 1.5] for i in link_prob_isomorphism]

    obv_congest_data={
        "tree_name":true_congest_data["tree_name"],
        "tree_vector":true_congest_data["tree_vector"],
        "route_matrix":true_congest_data["route_matrix"],
        "measure_time":true_congest_data["measure_time"],
        "K":true_congest_data["K"],
        "link_prob_isomorphism":true_congest_data["link_prob_isomorphism"],
        "link_prob_isomerism":true_congest_data["link_prob_isomerism"],
        "error_list":error_list,
        "isomorphism":{},
        "isomerism":{}
    }
    link_prob_isomorphism= true_congest_data["link_prob_isomorphism"]
    link_prob_isomerism= true_congest_data["link_prob_isomerism"]

    #生成同构概率观测路径的数据
    true_isomorphism_data=true_congest_data["isomorphism"]
    for i in range(len(link_prob_isomorphism)):
        path_true_data=true_isomorphism_data[str(i)]["path_measure_data"]
        path_obv_data=get_obv_matrix_isomorphism(path_true_data,error_list)
        obv_congest_data["isomorphism"][str(i)]={}
        obv_congest_data["isomorphism"][str(i)]["path_obv_data"]=path_obv_data
        obv_congest_data["isomorphism"][str(i)]["link_prob_list"]=true_isomorphism_data[str(i)]["link_prob_list"]



    #生成异构概率观测路径数据
    true_isomerism_data=true_congest_data["isomerism"]
    for i in range(len(link_prob_isomerism)):
        path_true_data=true_isomerism_data[str(i)]["path_measure_data"]
        path_obv_data=get_obv_matrix_isomerism(path_true_data,error_list)
        obv_congest_data["isomerism"][str(i)]={}
        obv_congest_data["isomerism"][str(i)]["path_obv_data"]=path_obv_data
        obv_congest_data["isomerism"][str(i)]["prob_scope"]=true_isomerism_data[str(i)]["prob_scope"]
        obv_congest_data["isomerism"][str(i)]["link_prob_list"]=true_isomerism_data[str(i)]["link_prob_list"]

    # print("生成的所有观测数据")
    # pprint(obv_congest_data)
    # print("调试")

    #保存数据到target数据
    print("保存数据开始"+target_path)
    with open(target_path,'w') as fw:
        json.dump(obv_congest_data,fw,indent=4)
    print("保存数据结束"+target_path)


def get_obv_matrix_isomorphism(path_true_data,error_list):
    """
    同构真实路径拥塞矩阵，根据观测误差列表生成 仿真的路径拥塞矩阵
    path_obv_data{
        obv_error: path_data_list (path,time)
        。。。
    }
    :param path_true_data: list(path,time) 真实的路径拥塞矩阵
    :param error_list:  list(n,) 观测误差列表
    :return:  path_obv_data dict
    """
    #最终生成的有误差的观测路径数据
    path_obv_data={}
    path_num=len(path_true_data)
    time=len(path_true_data[0])

    #根据均匀分布随机生成一个（path,time）随机矩阵
    path_obv_data_random=np.random.uniform(0,1,(path_num,time)).tolist()
    # print("随机生成的矩阵")
    # pprint(path_obv_data_random)
    #根据随机矩阵和错误率 跟换path_true_data中的真实结果，生成观测结果
    for i in range(len(error_list)):
        if i==0:
            path_obv_data[str(error_list[i])]=copy.deepcopy(path_true_data)
        else:
            temp_path_data = copy.deepcopy(path_true_data)
            for j in range(len(path_obv_data_random)):
                for k in range(len(path_obv_data_random[0])):
                    if path_obv_data_random[j][k]<=error_list[i]:
                        temp_path_data[j][k]=temp_path_data[j][k]^1   #和1异或 就是取反
            path_obv_data[error_list[i]]=copy.deepcopy(temp_path_data)



    return path_obv_data

def get_obv_matrix_isomerism(path_true_data,error_list):
    """
        异构真实路径拥塞矩阵，根据观测误差列表生成 仿真的路径拥塞矩阵
        path_obv_data{
            obv_error: path_data_list (K,path,time)
            。。。
        }
        :param path_true_data: list(k,path,time) 真实的路径拥塞矩阵
        :param error_list:  list(n,) 观测误差列表
        :return:  path_obv_data dict
        """
    # 最终生成的有误差的观测路径数据
    path_obv_data = {}
    K = len(path_true_data)
    path_num = len(path_true_data[0])
    time=len(path_true_data[0][0])

    # 根据均匀分布随机生成一个（path,time）随机矩阵
    path_obv_data_random = np.random.uniform(0, 1, (K,path_num, time)).tolist()
    # print("随机生成的矩阵")
    # pprint(path_obv_data_random)
    # 根据随机矩阵和错误率 跟换path_true_data中的真实结果，生成观测结果
    for i in range(len(error_list)):
        if i == 0:
            path_obv_data[str(error_list[i])] = copy.deepcopy(path_true_data)
        else:
            temp_path_data = copy.deepcopy(path_true_data)
            for j in range(K):
                for m in range(path_num):
                    for n in range(time):
                        if path_obv_data_random[j][m][n] <= error_list[i]:
                            temp_path_data[j][m][n] = temp_path_data[j][m][n] ^ 1  # 和1异或 就是取反
            path_obv_data[error_list[i]] = copy.deepcopy(temp_path_data)

    return path_obv_data



# ----------------------------------一下是测试代码---------------------
def test():
    # l = np.random.uniform(0, 0.5, 10)
    # print(l)
    # for i in range(10):
    #     l = i
    # print(l)

    # l=np.array([1,0,1,3,4])
    # l=np.where(l<1,l,1)
    # print(l)
    # l = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    # l_a = np.array(l)
    # print(l_a)
    #
    # l_sum = l_a.sum(axis=2)
    # print(l_sum)
    a=[1,2]
    c={}
    c["1"]=a
    a[0]=0
    print(a)
    print(c)

    pass


def test_gen_congest_isomorphism():
    tree_vector = [0, 1, 1, 3, 3]
    route_matrix = [[1, 1, 0, 0, 0],
                    [1, 0, 1, 1, 0],
                    [1, 0, 1, 0, 1]]
    time = 10
    prob = 0.1
    link_congest_M, path_congest_M = gen_congest_isomorphism(tree_vector, route_matrix, time, prob)
    print("输出链路拥塞结果")
    pprint(link_congest_M)
    print("输出路径拥塞结果")
    pprint(path_congest_M)

    link_sum = [sum(i) for i in link_congest_M]
    print("链路拥塞次数")
    print([i / time for i in link_sum])


def test_gen_congest_isomerism():
    """
    测试gen_congest_isomerism方法
    :return:
    """
    tree_vector = [0, 1, 1, 3, 3]
    route_matrix = [[1, 1, 0, 0, 0],
                    [1, 0, 1, 1, 0],
                    [1, 0, 1, 0, 1]]
    time = 5
    prob_scope = [0.1, 0.2]
    k = 2
    link_congest_MK, path_congest_MK, link_prob_list = gen_congest_isomerism(tree_vector, route_matrix, time,
                                                                             prob_scope, k)

    print("输出链路拥塞结果")
    pprint(link_congest_MK)
    print("输出路径拥塞结果")
    pprint(path_congest_MK)

    link_sum_k = []
    for i in range(k):
        link_sum = [sum(j) for j in link_congest_MK[i]]
        new_link_sum = [i / time for i in link_sum]
        link_sum_k.append(new_link_sum)
    print("输出所有的链路拥塞次数")
    pprint(link_sum_k)
    print("输出所有链路的概率组")
    pprint(link_prob_list)


def test_gen_a_congest_data():
    """
    测试gen_a_congest_data方法  生成
    :return:
    """
    tree_vector = [0, 1, 1, 3, 3]
    tree_name = "first_true_test"
    measure_time = 10
    K = 2
    link_prob_isomorphism = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    link_prob_isomerism = [[i * 0.5, i * 1.5] for i in link_prob_isomorphism]
    congest_data = gen_a_congest_data(tree_vector, tree_name, measure_time, K, link_prob_isomorphism,
                                      link_prob_isomerism)

    # 保存数据
    file_path = os.path.join(os.getcwd(), "test_data",tree_name + '.json')
    print("保存文件开始", file_path)
    with open(file_path, 'w')as fw:
        json.dump(congest_data, fw, indent=4)
    print("保存文件结束", file_path)

    print("输出测试结果")
    # pprint(congest_data)

    print("同构概率测试")
    link_sum_link_prob_isomorphism = []
    for i in range(len(link_prob_isomorphism)):
        link_measure_data = congest_data["isomorphism"][str(i)]["link_measure_data"]
        link_sum = [sum(i) for i in link_measure_data]
        new_link_sum = [i / measure_time for i in link_sum]
        link_sum_link_prob_isomorphism.append(new_link_sum)
    print("同构链路概率")
    print(link_prob_isomorphism)
    print("链路概率统计结果")
    pprint(link_sum_link_prob_isomorphism)

    for i in range(len(link_prob_isomorphism)):
        link_measure_data = congest_data["isomerism"][str(i)]["link_measure_data"]
        link_prob_list = congest_data["isomerism"][str(i)]["link_prob_list"]
        link_measure_data = np.array(link_measure_data)
        new_res = link_measure_data.sum(axis=2)
        new_res = new_res / measure_time

        print("异构" + str(i))
        print("链路概率")
        pprint(link_prob_list)
        print(("统计结果"))
        pprint(new_res)


def test_data():
    """
    测试生成json数据的正确性  Arpanet19706.json
    :return:
    """
    file_path = os.path.join(os.getcwd(), "data", "Arpanet19706.json")
    res = {}
    with open(file_path, "r") as fr:
        res = json.load(fr)

    pprint(res)

#------------------测试生成带误差的观测路径数据
def test_get_obv_matrix_isomorphism():
    """
    测试get_obv_matrix_isomorphism
    :return:
    """

    path_true_data=[[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]]
    error_list=[0.0,0.5,0.6]

    print("原矩阵")
    pprint(path_true_data)
    data=get_obv_matrix_isomorphism(path_true_data, error_list)
    print("同构链路生成的路径观测矩阵")
    pprint(data)

def test_get_obv_matrix_isomerism():
    """
    测试 get_obv_matrix_isomerism函数的正确性
    :return:
    """
    path_true_data = [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                      [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
    error_list = [0.0, 0.5, 0.6]
    print("原矩阵")
    pprint(path_true_data)
    data = get_obv_matrix_isomerism(path_true_data, error_list)
    print("同构链路生成的路径观测矩阵")
    pprint(data)

def test_gen_a_path_obv_data():
    """
    测试一个topo的观测路径矩阵生成
    从first_true_test.json 中读取数据 生成的数据保存到first_obv
    :return:
    """
    #读取test_data 中的数据
    file_path=os.path.join(os.getcwd(),"test_data","first_true_test.json")
    target_path=os.path.join(os.getcwd(),"test_data","first_obv_test.json")

    error_list=[0.0,0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    gen_a_path_obv_data(file_path,target_path,error_list)
    pass

if __name__ == '__main__':
    # gen_all_congest_data()
    # test()
    # test_gen_congest_isomorphism()
    # test_gen_congest_isomerism()
    # test_gen_a_congest_data()
    # gen_all_congest_data()

    # test_data()
    # test_get_obv_matrix_isomorphism()
    # test_get_obv_matrix_isomerism()
    # test_gen_a_path_obv_data()
    gen_all_path_obv_data()

    pass
