#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/24 18:15
# @Author  : Lelsey
# @File    : gen_dataset.py
#将datazoom和topoimgs，和topoinfo.csv中的文件保存到一起，在all_DS下的TOPO_INFO中
import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import itertools
import pandas as pd
import json
import shutil

"""
从topozoom中获取所有的topo，同时以json的格式保存
数据来源：
"""
def findAllFile(base):  #遍历文件夹中的某类文件
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.graphml'):
                fullname = os.path.join(root, f)
                yield fullname

def draw_graph_with_pos(data):  #绘制给定坐标的拓扑图，考虑到个别点未给的情况
        G = nx.read_graphml(data)
        x=[]
        y=[]
        i=0
        for n in G.nodes( data = True):
            if 'Latitude' in n[1].keys():#读取经纬度存在的点，并保存到两个列表
#               print(n)
                x.append(n[1]['Longitude'])
                y.append(n[1]['Latitude'])
            elif x:                     #未标注经纬度的点不是第一个点的情况
                x.append(random.uniform(min(x),max(x)))
                y.append(random.uniform(min(y),max(y)))
            else:                       #是前面的点时，则随机生成
                x.append(random.uniform(-50,50))
                y.append(random.uniform(-50,50))
        #    x[i]=n[1]['Latitude']
        #    y[i]=n[1]['Longitude']
        coordinates = []
        for i in range(G.number_of_nodes()):
            coordinates.append((x[i],y[i]))
        vnode= np.array(coordinates)
        npos = dict(zip(G.nodes, vnode))
        pos = {}
        pos.update(npos)#前面的x、y两个列表合并成坐标，存到pos
        plt.figure()
        nx.draw(G,pos, with_labels=True, node_size=500, node_color='red', node_shape='.')
        return G

def draw_graph_without_pos(data):#绘制未给坐标的拓扑图
    G=nx.read_graphml(data)
    pos = nx.spring_layout(G)#Layout是随机分布
    plt.figure()
    nx.draw(G,pos, with_labels=True, node_size=500, node_color='red', node_shape='.')
#    plt.show()#显示图像
    return G

def generate_all_topo():
    """
    对所有的gml文件进行操作读取相关信息
    :return:
    """
    #（1）从./topoinfo.csv中获取所有topo的基本信息
    topo_info_csv_path='./topoInfo.csv'
    all_topo_dict=read_csv(topo_info_csv_path)
    new_topo_dict={}

    #mesh_topo_info[(节点数量，链路数量，路径数，网络直径)]
    #tree_topo_info[(节点数量，链路数量，路径数,深度，广度)
    all_csv_content={
        'tree_name':[],
        'adj_matrix':[],
        '(mesh_nodes_edges_paths_diameter)':[],
        'tree_vector':[],
        '(tree_nodes_edges_paths_depth_breadth)':[],
        'image_name':[]
    }

    #(2)从all_nets/dataset_zoom中获取所有网络的信息
    plt.close('all')
    print("程序开始")
    for i in findAllFile('./dataset_zoom/'):  # 遍历文件，绘制给出经纬度的拓扑
        print("开始进行第" + str(i) + "个图")

        #获取网络名
        file_path = i
        (filepath, tempfilename) = os.path.split(file_path)  # 分离文件路径名
        (filename, extension) = os.path.splitext(tempfilename)  # 分离文件类型名
        print(filename)

        if filename not in all_topo_dict.keys():
            print("------------------------")
            print("没有",filename)
            print("---------------------------")
            continue
        try:
            #画图，保存数据
            g = draw_graph_with_pos(i)
            mesh_diameter = nx.diameter(g)
        except:
            continue
        else:
            # 生成数据目录
            package_path = os.path.dirname(os.getcwd())
            all_topo_path = os.path.join(package_path, 'all_DS', 'TOPO_INFO')
            a_topo_info_dir = os.path.join(all_topo_path, filename)
            if not os.path.exists(a_topo_info_dir):
                os.mkdir(a_topo_info_dir)
            a_topo_info_path = os.path.join(a_topo_info_dir, filename + '.png')

            plt.savefig(a_topo_info_path, format='png', dpi=200)
            #获取临接矩阵
            adj_matrix = np.array(nx.adjacency_matrix(g).todense())

            all_topo_dict[filename]["adj_matrix"]=adj_matrix.tolist()
            #获取路由矩阵
            route_matrix=tree_vector_to_route_matrix(all_topo_dict[filename]["tree_vector"])
            all_topo_dict[filename]["route_matrix"]=route_matrix.tolist()
            #获取mesh网络的基本信息
            mesh_num_nodes=g.number_of_nodes()
            mesh_num_edges=g.number_of_edges()
            num_paths=mesh_num_nodes*(mesh_num_nodes-1)
            mesh_diameter=nx.diameter(g)
            mesh_topo_info=(mesh_num_nodes,mesh_num_edges,num_paths,mesh_diameter)
            all_csv_content["tree_name"].append(filename)
            all_csv_content["adj_matrix"].append(adj_matrix.tolist())
            all_csv_content["(mesh_nodes_edges_paths_diameter)"].append(mesh_topo_info)

            #获取生成树的相关信息
            tree_vector=all_topo_dict[filename]["tree_vector"]
            tree_num_nodes=all_topo_dict[filename]["link_numbers"]+1
            tree_num_links=all_topo_dict[filename]["link_numbers"]
            tree_num_paths=all_topo_dict[filename]["path_numbers"]
            tree_depth=all_topo_dict[filename]["tree_depth"]
            tree_breadth=cal_tree_breadth(tree_vector,tree_depth)
            tree_csv_info=(tree_num_nodes,tree_num_links,tree_num_paths,tree_depth,tree_breadth)
            image_name= all_topo_dict[filename]["file_name"]
            all_csv_content["tree_vector"].append(tree_vector)
            all_csv_content["(tree_nodes_edges_paths_depth_breadth)"].append(tree_csv_info)
            all_csv_content["image_name"].append(image_name)




            #拷贝生成树文件
            image_name=all_topo_dict[filename]["file_name"].replace('_','-')
            old_topo_image_path=os.path.join(package_path,"all_nets","topoImgs",image_name)
            new_topo_image_path=os.path.join(package_path,"all_DS","TOPO_INFO",filename,all_topo_dict[filename]["file_name"])
            shutil.copy(old_topo_image_path,new_topo_image_path)

            #将每一个topo的信息存储带对应的文件夹中 保存到json文件
            topo_json_file=os.path.join(a_topo_info_dir,filename+'.json')
            with open(topo_json_file,'w') as fw:
                json.dump(all_topo_dict[filename],fw,indent=4)
            # print(all_topo_dict[filename])

            #记录所有topo的信息 保存到json文件中
            a_topo_dict={
                "name":filename,
                "tree_vector":all_topo_dict[filename]["tree_vector"],
                "file_name":all_topo_dict[filename]["file_name"]
            }
            new_topo_dict[filename]=a_topo_dict


    #保存网络的信息到csv文件
    data_frame=pd.DataFrame(all_csv_content)
    all_topo_csv_path = os.path.join(os.path.dirname(os.getcwd()), 'all_DS', 'TOPO_INFO', "all_topo.csv")
    data_frame.to_csv(all_topo_csv_path)
    all_topo_excel_path = os.path.join(os.path.dirname(os.getcwd()), 'all_DS', 'TOPO_INFO', "all_topo.xlsx")
    writer = pd.ExcelWriter(all_topo_excel_path, engine="xlsxwriter")
    data_frame.to_excel(writer)
    writer.save()
    #保存所有网络的信息
    all_topo_json_path = os.path.join(os.path.dirname(os.getcwd()), 'all_DS', 'TOPO_INFO',"all_topo.json")
    with open(all_topo_json_path,'w') as fw:
        json.dump(new_topo_dict,fw,indent=4)



    print("程序结束")
    pass

def tree_vector_to_route_matrix(tree_vector:list):
    """
    将树向量变为路由矩阵
    :param tree_vector: 树向量
    :return: route_matrix array(m,n) 路由矩阵
    """

    leaf_nodes = []
    for index in range(len(tree_vector)):
        leaf_node = index + 1
        if leaf_node not in tree_vector:
            leaf_nodes.append(leaf_node)

    num_paths=len(leaf_nodes)
    num_links=len(tree_vector)

    route_matrix = np.zeros((num_paths, num_links), dtype=int)
    for i in range(num_paths):
        leaf_node = leaf_nodes[i]
        route_matrix[i][leaf_node - 1] = 1
        parent_node = tree_vector[leaf_node - 1]
        while parent_node != 0:
            route_matrix[i][parent_node - 1] = 1
            parent_node = tree_vector[parent_node - 1]
    return route_matrix

def read_csv(file_path:str):
    """
    读取cvs文件中的信息
    :param file_path: 文件路径
    :return:
    """

    csv_reader = csv.reader(open(file_path))
    all_line=[]
    for line in csv_reader:
        all_line.append(line)
    # print("print网络所有信息")
    # for i in all_line:
    #     print(i)

    all_topo={}
    for line in all_line[1:]:

        a_topo={}
        a_topo["topo_name"]=line[1]
        a_topo["tree_vector"]=eval(line[2])
        a_topo["link_numbers"]=int(line[3])
        a_topo["path_numbers"]=int(line[4])
        a_topo["tree_depth"]=int(line[5])
        a_topo["file_name"]=line[6]
        all_topo[line[1]]=a_topo
    # print("csv文件中的数据")
    # print(all_topo)

    return all_topo


def cal_tree_breadth(tree_vector:list,depth):
    """
    计算树的宽度
    :param tree_vector:
    :param depth:
    :return:
    """
    level=[0 for i in range(depth+1)]
    for i in range(len(tree_vector)):
        node_level=1
        node_index=i
        while tree_vector[node_index]!=0:
            node_level+=1
            node_index=tree_vector[node_index]-1
        level[node_level]+=1
    # print(level)
    return max(level)


def test_cal_tree_breadth():
    tree_vector=[0,1,1,3,3,5,5,6,6,6]
    cal_tree_breadth(tree_vector,5)

def test_read_csv():
    """

    :return:
    """
    file_path = "./topoInfo.csv"
    read_csv(file_path)

def test():
    a=[1,2,3]
    print(a[1:])
    b="[1,2,3]"
    c=eval(b)
    print(c)

def test1():
    file_path=os.path.join(os.path.dirname(os.getcwd()),'all_DS',"TOPO_INFO")
    print(file_path)
    res=os.listdir(file_path)
    print(res)
    print(len(res))

    file_path1=os.path.join(os.getcwd(),"dataset_zoom")
    res2=os.listdir(file_path1)
    print(len(res2))

def test_excel():
    #测试excel文件
    data={
        "a":[1,2,3],
        "b":[1,2,3],
    }
    data_frame=pd.DataFrame(data)
    writer=pd.ExcelWriter("test.xlsx",engine="xlsxwriter")
    data_frame.to_excel(writer)
    writer.save()
    # data_frame.to_excel("test.xls",index=False)

if __name__ == '__main__':

    # test()
    generate_all_topo()
    # test1()
    # test_cal_tree_breadth()
    # test_excel()
    pass