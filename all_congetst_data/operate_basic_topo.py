#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/21 20:42
# @Author  : Lelsey
# @File    : operate_basic_topo.py

import pandas as pd
import os
import numpy as np
from all_tools import utils
import json
from pprint import pprint


def save_basic_topo_data(filepath):
    """
    读取xlsx文件，保存数据到all_basic_topo.json中
    :param filepath: str
    :return:
    """

    # 从xlsx中读取数据
    dframe = pd.read_excel(filepath)
    data_dict = dframe.to_dict("list")
    # print(dframe)

    # mesh_topo_info[(节点数量，链路数量，路径数，网络直径)]
    # tree_topo_info[(节点数量，链路数量，路径数,深度，广度)
    all_csv_content = {
        'tree_name': [],
        'adj_matrix': [],
        '(mesh_nodes_edges_paths_diameter)': [],
        'tree_vector': [],
        '(tree_nodes_edges_paths_depth_breadth)': [],
        'image_name': []
    }

    # 将dframe中每一列数据保存到list中
    all_csv_content['tree_name'] = data_dict['tree_name']
    all_csv_content["tree_vector"] = data_dict['tree_vector']
    all_csv_content['(tree_nodes_edges_paths_depth_breadth)'] = data_dict['(tree_nodes_edges_paths_depth_breadth)']
    all_csv_content['image_name'] = data_dict['image_name']

    # 将每一个拓扑的属性保存到字典中
    all_basic_topo = {}
    for i in range(len(all_csv_content['tree_name'])):
        a_basic_topo = {
            'name': '',
            'graph_name': '',
            'tree_vector': [],
            'route_matrix': [],
            'node_num': 0,
            'path_num': 0,
            'depth': 0,
            'breadth': 0,
            'path_length_list': [],
            'leaf_list': []
        }

        a_basic_topo['name'] = all_csv_content['tree_name'][i]
        a_basic_topo['graph_name'] = all_csv_content['image_name'][i]
        #tree_vector中字符转数字
        a_basic_topo['tree_vector'] = all_csv_content['tree_vector'][i][1:-1].split(',')
        a_basic_topo['tree_vector']=[int(i) for i in a_basic_topo['tree_vector']]
        a_basic_topo['route_matrix'] = utils.tree_vector_to_route_matrix(a_basic_topo['tree_vector']).tolist()

        tree_nodes_edges_paths_depth_breadth = all_csv_content['(tree_nodes_edges_paths_depth_breadth)'][i]
        tree_nodes_edges_paths_depth_breadth = tree_nodes_edges_paths_depth_breadth[1:-1].split(',')
        a_basic_topo['node_num'] = int(tree_nodes_edges_paths_depth_breadth[0])

        a_basic_topo['path_num'] = len(a_basic_topo['route_matrix'])
        a_basic_topo['depth'] = int(tree_nodes_edges_paths_depth_breadth[3])
        a_basic_topo['breadth'] = int(tree_nodes_edges_paths_depth_breadth[4])

        a_basic_topo['path_length_list'] = get_path_length_list(a_basic_topo['route_matrix'])
        a_basic_topo['leaf_list'] = get_leaf_list(a_basic_topo['route_matrix'])

        all_basic_topo[a_basic_topo['name']] = a_basic_topo

    # 保存数据到json文件
    json_path = os.path.join(os.getcwd(), "all_basic_topo.json")
    print("保存开始文件到" + json_path)
    with open(json_path, "w") as fw:
        json.dump(all_basic_topo, fw, indent=4)
    print("保存文件结束")



def get_path_length_list(route_matrix):
    """
    根据路由矩阵获取 每一个路径的长度
    :param route_matrix: list(path,link)
    :return:  path_length_list list(path,)
    """
    return [sum(i) for i in route_matrix]


def get_leaf_list(route_matrix):
    """
    根据路由矩阵获取 ，每一条路径的最终叶子结点
    :param route_matrix: list(path,link)
    :return: leaf_list list(path,)
    """
    route_matrix_shape = np.array(route_matrix).shape
    leaf_list = []
    for i in range(route_matrix_shape[0]):
        leaf = 0
        for j in range(route_matrix_shape[1]):
            if route_matrix[i][j] == 1:
                leaf = j
        leaf += 1
        leaf_list.append(leaf)
    return leaf_list


def statistics_data():
    """
    统计all_basic_topo.json中的数据
    :return:
    """
    file_path=os.path.join(os.getcwd(),"all_basic_topo.json")
    all_data={}
    with open(file_path,'r') as fr:
        all_data=json.load(fr)

    #将所有的topo中的数据每一项存储到列表中
    name_list=[]
    node_num_list=[]
    path_num_list=[]
    depth_list=[]
    breadth_list=[]
    all_path_length_list=[]

    for key,value in all_data.items():
        a_basic_topo=value

        name_list.append(a_basic_topo['name'])
        node_num_list.append(a_basic_topo['node_num'])
        path_num_list.append(a_basic_topo['path_num'])
        depth_list.append(a_basic_topo['depth'])
        breadth_list.append(a_basic_topo['breadth'])
        all_path_length_list.append(sum(a_basic_topo['path_length_list']))

    print(name_list)
    statis_data={
        "node_num":{},
        "path_num":{},
        "depth":{},
        "breadth":{},
        "path_length":{}
    }
    statis_data["node_num"]=cal_max_min_even_var_std_frequency(node_num_list)
    statis_data["path_num"]=cal_max_min_even_var_std_frequency(path_num_list)
    statis_data["depth"]=cal_max_min_even_var_std_frequency(depth_list)
    statis_data["breadth"]=cal_max_min_even_var_std_frequency(breadth_list)
    statis_data["path_length"]=cal_max_min_even_var_std_frequency(all_path_length_list)

    print("输出所有的统计数据")
    pprint(statis_data)


    out_statis_data(statis_data["node_num"],5,53,"节点数量")
    out_statis_data(statis_data["breadth"],2,20,"树的宽度")
    out_statis_data(statis_data["depth"],2,10,"树的深度")
    out_statis_data(statis_data["path_num"],2,42,"路径数量")
    out_statis_data(statis_data["path_length"],6,153,"路径长度")




def out_statis_data(data,left,right,name):
    """
    格式化输入关于拓扑属性的数据
    :param data:  数据文件 dict
    :param left:  区间左端点
    :param right:  区间右端点
    :param name:   数据属性
    :return:
    """
    print()
    print("{}统计数据----------------------------".format(name))
    print("最大值", data["data_max"], "最小值", data["data_min"], "平均值",
          data["data_even"],
          "方差", data["data_var"], "标准差",data["data_std"])

    print("区间分布")

    all_sum = 0
    for kv in data["data_frequency"].items():
        if kv[0] >= left and kv[0] <= right:
            all_sum = all_sum + kv[1]
    print("{}在 {} 和 {} 之间的占比是 {}".format(name,left, right, all_sum))



def cal_max_min_even_var_std_frequency(data_list):
    """
    统计data_list中的最大值，最小值，平均值，方差，频率分布
    :param data_list: list(n,)
    :return: info
    """
    data_max=max(data_list)
    data_min=min(data_list)
    data_array=np.array(data_list)
    data_even=np.mean(data_array)
    data_var=np.var(data_array)
    data_std=np.std(data_array)

    #统计频率
    pd_data=pd.Series(data_list)
    data_frequency = dict(pd_data.value_counts(sort=False, normalize=True))

    info={
        "data_max":data_max,
        "data_min":data_min,
        "data_even":data_even,
        "data_var":data_var,
        "data_std":data_std,
        "data_frequency":data_frequency
    }

    return info





#===================================以下是测试文件

def test_path_leaf():
    """
    测试获取路径长度类表和叶子结点列表
    :return:
    """
    # route_matrix = [[1, 1, 0, 0, 0],
    #                 [1, 0, 1, 1, 0],
    #                 [1, 0, 1, 0, 1]]

    route_matrix = [[1, 1, 0, 1, 0, 0, 0],
                    [1, 1, 0, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 1]]

    path_length_list = get_path_length_list(route_matrix)
    leaf_list = get_leaf_list(route_matrix)
    print("path_length_list", path_length_list)
    print("leaf_list", leaf_list)


def test_save_all_data():
    filepath = os.path.join(os.getcwd(), "all_topo.xlsx")
    save_basic_topo_data(filepath)

def test_frequency():
    ls=[3,2,1,3,2,1,3]
    se=pd.Series(ls)

    countDict = dict(se.value_counts(sort=False,normalize=True))
    proportitionDict = dict(se.value_counts(normalize=True))

    print(countDict)
    print(proportitionDict)

def test_cal_max_min_even_var_std_frequency():
    l=[1,2,3,1,2,3,1]
    info=cal_max_min_even_var_std_frequency(l)
    print("最大值",info["data_max"])
    print("最小值",info["data_min"])
    print("平均值",info["data_even"])
    print("方差",info["data_var"])
    print("标准差",info["data_std"])
    print("频率分布",info['data_frequency'])


if __name__ == '__main__':
    # test_save_all_data()
    # test_path_leaf()
    # statistics_data()
    # test_frequency()
    # test_cal_max_min_even_var_std_frequency()
    statistics_data()