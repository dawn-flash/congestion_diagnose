#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/6 15:14
# @Author  : Lelsey
# @File    : motivation.py

"""
同构异构的8组图。横轴观测误差，纵轴（DR，FPR，F1，F2）（比例：6:8）双栏，一组4个图。字10.5pt  ，英文字体times，中文宋体，边框调粗
拓扑规模（变成表，统计性能下降情况，量度待定，12个拓扑）
拓扑选择：

拓扑：bellcanada,
+  [0, 1, 1, 1, 1, 1,  3, 3, 3, 5, 5, 6, 6, 6, 9, 9, 12, 12, 15, 15, 15, 17, 17, 22, 22, 25, 25, 25,  27, 27, 27]
+ 32个节点 20条path
"""

import os
import numpy as np
import json
from pprint import pprint
from new_algs.alg_clink import alg_clink
from new_algs.alg_scfs import alg_scfs
from new_algs.alg_map_advace import alg_map_advace
from all_tools import utils


def read_data(true_data_path,obv_data_path):
    """
    读取拓扑的拥塞数据，返回真实的数据和观测的数据
    :param topo_name  str  拓扑的名字
    :return: true_data,obv_data   :dict  topo_name.json中的真实数据和观测数据
    """
    true_data={}
    obv_data={}

    print("{}开始读取------------".format(true_data_path))
    with open(true_data_path,"r") as fr:
        true_data=json.load(fr)

    print("{}开始读取------------".format(obv_data_path))
    with open(obv_data_path,"r") as fr:
        obv_data=json.load(fr)

    print("数据读取完成")

    # print("true_data")
    # print(true_data)
    # print("obv_data")
    # print(obv_data)
    return true_data,obv_data

def save_data(file_path,result_data):
    """
    保存文件数据到file_path中
    :param file_path: 保存路径
    :param result_data : dict 保存的数据
    :return:
    """



    print("保存数据到{}开始".format(file_path))
    with open(file_path,'w')as fw:
        json.dump(result_data,fw,indent=4)
    print("保存数据结束")

def isomorphism_experiment(true_data_path,obv_data_path,file_save_path):
    """
    生成同构的四组图数据， topo：bellcanada,
    link_prob_isomorphism = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    路径的观测误差[0.0，0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    横轴：观测误差
    纵轴：DR，FPR，F1，F2
    time=1000
    K=20
    :return:
    """
    #读取文件数据
    true_data,obv_data=read_data(true_data_path,obv_data_path)

    #获取基本实验信息
    tree_name=obv_data["tree_name"]
    measure_time=true_data["measure_time"]
    error_list=obv_data["error_list"]
    error_list=[str(i) for i in error_list]

    #目前挑选同构拥塞概率为0.1
    link_prob_isomorphism="0.1"
    link_prob_isomorphism_index="5"
    Route_Matrix=np.array(obv_data["route_matrix"])

    #获取观测路径信息，真实链路概率
    isomorphism_data=obv_data["isomorphism"][link_prob_isomorphism_index]
    path_obv_data=isomorphism_data["path_obv_data"]
    link_prob_list=np.array(isomorphism_data["link_prob_list"])

    #获取真实的链路拥塞状态
    link_measure_data   =np.array(true_data["isomorphism"][link_prob_isomorphism_index]["link_measure_data"])


    #算法链路推测结果为numpy(link,time)
    result_data={
        "scfs":{},
        "clink":{},
        "map":{},
        "error_list":error_list,
        "link_prob_isomorphism":link_prob_isomorphism,
        "link_prob_list":link_prob_list.tolist(),
        "measure_time":measure_time,
        "link_measure_data":link_measure_data.tolist(),
        "evaluate_data":{
            "dr":{
                "scfs":[],
                "clink":[],
                "map":[]
            },
            "fpr":{
                "scfs": [],
                "clink": [],
                "map": []
            },
            "f1":{
                "scfs": [],
                "clink": [],
                "map": []
            },
            "f2":{
                "scfs": [],
                "clink": [],
                "map": []
            },
        }
    }

    for i in range(len(error_list)):
        a_group_path_obv=np.array(path_obv_data[error_list[i]])

        scfs_link_infer=np.array([])
        clink_link_infer=np.array([])
        map_link_infer=np.array([])

        #执行算法
        print("路径的误差概率为{}----------开始".format(error_list[i]))
        print("scfs开始执行")
        scfs_link_infer=alg_scfs(a_group_path_obv,Route_Matrix)
        print("clink开始执行")
        clink_link_infer=alg_clink(a_group_path_obv,Route_Matrix,link_prob_list)
        print("map开始执行")
        map_link_infer=alg_map_advace(a_group_path_obv,Route_Matrix,link_prob_list)
        print("路径的误差概率为{}----------结束".format(error_list[i]))


        #保存推测结果到result_data
        result_data["scfs"][i]=scfs_link_infer.tolist()
        result_data["clink"][i]=scfs_link_infer.tolist()
        result_data["map"][i]=scfs_link_infer.tolist()

        # 评估计算推断结果 （dr，fpr，f1,f2）
        scfs_evalute=utils.get_drfpr_f1j(scfs_link_infer,link_measure_data,1,"averaged",False)
        clink_evalute=utils.get_drfpr_f1j(clink_link_infer,link_measure_data,1,"averaged",False)
        map_evalute=utils.get_drfpr_f1j(map_link_infer,link_measure_data,1,"averaged",False)

        # print("scfs_link_infer")
        # pprint(scfs_link_infer)
        # print("link_measure_data")
        # pprint(link_measure_data)

        #保存评估结果
        #保存dr
        result_data["evaluate_data"]["dr"]["scfs"].append(scfs_evalute[0])
        result_data["evaluate_data"]["dr"]["clink"].append(clink_evalute[0])
        result_data["evaluate_data"]["dr"]["map"].append(map_evalute[0])

        # 保存fpr
        result_data["evaluate_data"]["fpr"]["scfs"].append(scfs_evalute[1])
        result_data["evaluate_data"]["fpr"]["clink"].append(clink_evalute[1])
        result_data["evaluate_data"]["fpr"]["map"].append(map_evalute[1])

        # 保存f1
        result_data["evaluate_data"]["f1"]["scfs"].append(scfs_evalute[2])
        result_data["evaluate_data"]["f1"]["clink"].append(clink_evalute[2])
        result_data["evaluate_data"]["f1"]["map"].append(map_evalute[2])

        # 保存f2
        result_data["evaluate_data"]["f2"]["scfs"].append(scfs_evalute[4])
        result_data["evaluate_data"]["f2"]["clink"].append(clink_evalute[4])
        result_data["evaluate_data"]["f2"]["map"].append(map_evalute[4])

    print("所有算法的结果")
    pprint(result_data["evaluate_data"])

    #保存数据
    save_data(file_save_path,result_data)


def Bellcanada_isomorphism_experiment():
    # Bellcanada 进行测试 文保存 experiment\experiment_res\motivation_res\Bellcanada_isomorphism.json
    file_dir = os.path.join(os.path.dirname(os.getcwd()), "all_congetst_data")
    true_data_path = os.path.join(file_dir, "data_true", "Bellcanada.json")
    obv_data_path = os.path.join(file_dir, "data_obv", "Bellcanada.json")
    file_save_path = os.path.join(os.getcwd(), "experiment_res", "motivation_res", "Bellcanada_isomorphism.json")
    isomorphism_experiment(true_data_path, obv_data_path, file_save_path)





#-------------------------------------test---------------------------------
def test():
    print(os.getcwd())
    print(os.path.join(os.path.dirname(os.getcwd()),"test_data"),)

def test_read_data():
    #测试read_data文件
    file_dir = os.path.join(os.path.dirname(os.getcwd()), "all_congetst_data", "test_data")
    true_data_path = os.path.join(file_dir, "first_true_test.json")
    obv_data_path = os.path.join(file_dir, "first_obv_test.json")
    read_data(true_data_path,obv_data_path)

def test_isomorphism_experiment():
    """
    对isomorphism_experiment 函数进行测试
    :return:
    """

    #测试一：使用teste.json文件进行测试
    # file_dir = os.path.join(os.path.dirname(os.getcwd()), "all_congetst_data", "test_data")
    # true_data_path = os.path.join(file_dir, "first_true_test.json")
    # obv_data_path = os.path.join(file_dir, "first_obv_test.json")
    # file_save_path=os.path.join(os.getcwd(),"experiment_res","test","first_isomorphism_test.json")
    # isomorphism_experiment(true_data_path,obv_data_path,file_save_path)
    pass







if __name__ == '__main__':

    # test()
    # test_read_data()
    test_isomorphism_experiment()
