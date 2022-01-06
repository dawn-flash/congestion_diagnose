#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 15:44
# @Author  : Lelsey
# @File    : cong_net.py
import numpy as np
import random
import math
from all_tools import utils
import os
import json
import copy

class Cong_net(object):
    #公共方法————————————————————————————————————————————————————

    def __init__(self):
        """
        该网络类的所有属性
        self.tree_vector:list  树向量
        self.leaf_nodes:ndarray   所有的叶子节点数组
        self.num_paths:int  路径数量
        self.num_links:int  链路数量
        self.route_matrix:ndarray  路由矩阵
        self.depth:int  树的深度

        链路拥塞相关属性
        self.links_cong_pro:ndarray  链路拥塞概率数组
        self.loss_model = loss_model   丢包率模型
        self.threshold = threshold   丢包率门限

        self.links_state_true:ndarray    真实链路拥塞状态数组
        self.links_loss_rate:ndarray   链路丢包率数组
        self.links_cong_true:ndarray    真实拥塞链路编号数组

        路径拥塞相关属性
        self.paths_loss_rate:ndarray   路径丢包率数组
        self.paths_state_true:ndarray  真实路径拥塞状态数组
        self.paths_cong_true:ndarray  真实拥塞路径编号数组
        self.paths_no_cong_true:ndarray  真实非拥塞路径编号数组

        self.paths_state_obv:ndarray 观测路径拥塞状态数组
        self.paths_cong_obv:ndarray  观测拥塞路径编号数组
        self.paths_no_cong_obv:ndarray  观测非拥塞路径编号数组
        """

        #使用两个初始化方法对类进行初始化
        #（1）从json中读取数据 进行类初始化
        #（2）根据链路拥塞概率生成数据，进行类初始化
        pass

    #使用从json文件中读取数据进行初始化的方法
    def json_init(self,basic_info:dict,cong_data_info:dict,cong_config_info:dict):
        """
        从json文件中将文件导入
        :param basic_info:dict   topo相关信息
        :param cong_data_info:dict    一组拥塞数据相关信息
        :param cong_config_info:dict  拥塞数据配置相关信息
        :return:
        """
        #topo基本信息配置
        self.tree_vector=basic_info["tree_vector"]
        self.leaf_nodes=basic_info["leaf_nodes"]
        self.num_paths=basic_info["num_paths"]
        self.num_links=basic_info["num_links"]
        self.route_matrix=np.array(basic_info["route_matrix"])
        self.depth=basic_info["depth"]

        #链路拥塞相关属性
        self.links_cong_pro=np.array(cong_config_info["links_cong_pro"])
        self.loss_model =np.array(cong_config_info["loss_model"])
        self.threshold = np.array(cong_config_info["threshold"])

        #链路相关信息
        self.links_state_true=np.array(cong_data_info["links_state_true"])
        self.links_loss_rate=np.array(cong_data_info["links_loss_rate"])
        self.links_cong_true=np.array(cong_data_info["links_cong_true"])


        #真实路径相关属性
        self.paths_loss_rate=np.array(cong_data_info["paths_loss_rate"])
        self.paths_state_true=copy.deepcopy(np.array(cong_data_info["paths_state_true"]))
        self.paths_cong_true=np.array(cong_data_info["paths_cong_true"])
        self.paths_no_cong_true=np.array(cong_data_info["paths_no_cong_true"])

        #观测的路径相关属性
        self.paths_state_obv=copy.deepcopy(np.array(cong_data_info["paths_state_obv"]))
        self.paths_cong_obv=np.array(cong_data_info["paths_cong_obv"])
        self.paths_no_cong_obv=np.array(cong_data_info["paths_no_cong_obv"])

    #数据自动生成初始化相关的方法
    def auto_init(self,tree_vector:list,loss_model:str,threshold:float,links_cong_pro=None):
        """
        根据概率自动生成网络相关数据
        :param tree_vector: list 树向量
        :param loss_model: str 丢包率模型
        :param threshold:  float 丢包率门限
        :param links_cong_pro: list 链路拥塞概率
        :return:
        """
        self.gen_base_topo_info(tree_vector)

        self.set_link_cong_pro(links_cong_pro)
        self.gen_cong_topo_info(loss_model,threshold)
        pass


    def set_link_cong_pro(self,link_cong_pro):
        """
        为所有链路设置链路的拥塞概率：
        :param link_cong_pro: float 设置为同构链路 ，list设置为异构链路
        :return:links_cong_pro:list
        """
        if isinstance(link_cong_pro,float):
            self.links_cong_pro = [link_cong_pro] * self.num_links  # 同构链路拥塞概率

        if isinstance(link_cong_pro,list):  #异构链路的拥塞概率
            self.links_cong_pro=link_cong_pro
        self.links_cong_pro=np.array(self.links_cong_pro)

    def gen_base_topo_info(self,tree_vector:list):
        """
        根据树向量，生成基本的网络信息
        :param tree_vector:
        :return:
        """
        self.tree_vector = tree_vector  # 树向量
        self.leaf_nodes = self.__get_leaf_nodes()  # 所有的叶子节点
        self.num_paths = self.__get_num_paths()  # 路径数量
        self.num_links = self.__get_num_links()  # 链路数量
        self.route_matrix = self.__get_route_matrix()  # 路由矩阵
        self.depth = self.__get_depth()  # 树的深度
        pass

    def gen_cong_topo_info(self,loss_model:str,threshold:float):
        """

        :param loss_model:
        :param threshold:
        :return:
        """

        self.loss_model = loss_model  # 丢包率模型
        self.threshold = threshold  # 丢包率门限

        # 生成链路拥塞状态列表，和链路丢包率列表
        self.links_state_true = self.__gen_links_cong_statu()  # 各链路拥塞状态
        self.links_loss_rate = self.__gen_links_loss_rate()  # 各链路丢包率
        # 求真实拥塞链路编号列表
        self.links_cong_true = self.__cal_cong_link_info(self.links_state_true)

        # 计算真实路径丢包率和，真实路径状态
        self.paths_loss_rate, self.paths_state_true = self.__init_path()  # 各路径丢包率、各路径真实拥塞状态
        # 求真实拥塞路径编号列表和真实非拥塞路径编号列表
        self.paths_cong_true,self.paths_no_cong_true=self.__cal_cong_path_info(self.paths_state_true)


        #计算观测路径状态
        self.paths_state_obv = self.__observe_path()  # 各路径观测拥塞状态
        #计算观测拥塞路径编号列表和观测非拥塞路径编号列表
        self.paths_cong_obv,self.paths_no_cong_obv=self.__cal_cong_path_info(self.paths_state_obv)



    #对topo进行操作的相关方法
    def get_children(self, node):
        """
        获取指定节点的所有子节点
        :param node: 指定的结点
        :return: children:list
        """
        children = []
        if node not in self.leaf_nodes:
            for index, item in enumerate(self.tree_vector):
                if item == node:
                    children.append(index + 1)
        return children

    def get_paths(self, link: int):
        """
        获取经过指定链路的所有路径。

        在路由矩阵中，第 0 列代表链路 1，第 1 列代表链路 2。依次类推。
        第 0 行代表路径 1，第 1 行代表路径 2。依次类推。
        :param link: 链路的编号
        :return:
        """
        assert link > 0
        paths, = np.where(self.route_matrix[:, link-1] > 0)
        return paths.tolist()

    def get_links(self, path: int):
        """
        获取指定路径经过的所有链路。
        :param path: 路径编号
        :return: links:list
        """
        assert path > 0
        links, = np.where(self.route_matrix[path-1, :] > 0)
        return (links + 1).tolist()

    #私有方法————————————————————————————————————————————————————

    def __gen_links_cong_statu(self):
        """
        根据指定链路拥塞概率生成链路拥塞状态
        :return:links_state:list
        """
        links_state = []
        for i in range(self.num_links):
            th = random.random()
            if th <= self.links_cong_pro[i]:  # 链路发生拥塞
                cong_status = 1
            else:  # 链路处于正常状态
                cong_status = 0
            links_state.append(cong_status)
        return np.array(links_state)

    def __gen_links_loss_rate(self):
        """
        根据链路的拥塞状态，生成链路丢包率
        :return: links_loss_rate:list
        """
        links_loss_rate = []
        for i in range(self.num_links):
            if int(self.links_state_true[i]) == 1:
                links_loss_rate.append(self.__gen_link_loss_rate_cong())
            else:
                links_loss_rate.append(self.__gen_link_loss_rate_good())
        return np.array(links_loss_rate)

    def __gen_link_loss_rate_cong(self):
        """
        生成拥塞链路丢包率
        :return: links_loss_rate:list
        """
        if self.loss_model == "loss_model_2":
            return 1.0 - 0.99 * np.random.power(50)
        elif self.loss_model == "loss_model_1":
            return np.random.uniform(0.01, 0.05)
        else:
            raise Exception("no such loss model")

    def __gen_link_loss_rate_good(self):
        """
        生成非拥塞链路丢包率
        :return: links_loss_rate:list
        """
        if self.loss_model == "loss_model_2":
            return 0.01 - 0.01 * np.random.power(3)
        elif self.loss_model == "loss_model_1":
            return np.random.uniform(0.0, 0.01)
        else:
            raise Exception("no such loss model")

    def __init_path(self):
        """
        获取路径的真实丢包率列表
        获取路径的真实拥塞状态列表
        :return:
        """
        path_loss_rate_list = []
        path_cong_status_list = []
        for i in range(self.num_paths):
            transmission_rate = 1.0
            cong_status = 0
            for j in range(self.num_links):
                if int(self.route_matrix[i][j]) == 1:
                    transmission_rate *= (1.0 - self.links_loss_rate[j])
                    cong_status = cong_status or self.links_state_true[j]

            path_loss_rate = 1.0 - transmission_rate
            path_cong_status = cong_status
            path_loss_rate_list.append(path_loss_rate)
            path_cong_status_list.append(path_cong_status)
        return np.array(path_loss_rate_list), np.array(path_cong_status_list)

    def __observe_path(self):
        #获取观测的路径状态
        path_states = []
        for i in range(self.num_paths):
            if self.loss_model == "loss_model_2":
                if self.paths_loss_rate[i] >= (1.0 - math.pow((1 - self.threshold), sum(self.route_matrix[i]))):
                    path_state = 1
                else:
                    path_state = 0
            elif self.loss_model == "loss_model_1":
                if self.paths_loss_rate[i] >= self.threshold:
                    path_state = 1
                else:
                    path_state = 0
            else:
                raise Exception("no such loss model")
            path_states.append(path_state)
        return np.array(path_states)

    def __cal_cong_path_info(self,paths_state:list):
        """
        根据路径状态，计算拥塞和非拥塞的路径编号
        :param paths_state:
        :return: paths_cong:list,paths_no_cong:list
        """
        paths_cong=[]
        paths_no_cong=[]
        for index in range(len(paths_state)):
            if int(paths_state[index]) == 1:
                # if int(self.path_states[index]) == 1:
                paths_cong.append(index + 1)
            else:
                paths_no_cong.append(index + 1)
        return np.array(paths_cong),np.array(paths_no_cong)

    def __cal_cong_link_info(self,links_state):
        """
        根据链路状态计算拥塞和非拥塞链路编号
        :param links_state: list
        :return: congested_links:list
        """
        congested_links = []
        for index in range(len(links_state)):
            if int(links_state[index]) == 1:
                congested_links.append(index + 1)
        return np.array(congested_links)


    def __get_leaf_nodes(self):
        """
        获取叶子结点列表
        :return: leaf_nodes:list
        """
        leaf_nodes = []
        for index in range(len(self.tree_vector)):
            leaf_node = index + 1
            if leaf_node not in self.tree_vector:
                leaf_nodes.append(leaf_node)
        return np.array(leaf_nodes)

    def __get_num_paths(self):
        """
        获取路径数量
        :return: num_paths:int
        """
        return len(self.leaf_nodes)

    def __get_num_links(self):
        """
        获取链路数量
        :return: num_links:int
        """
        return len(self.tree_vector)

    def __get_route_matrix(self):
        """
        获取路由矩阵(path ,link)
        :return: route_matrix:ndarray
        """
        route_matrix = np.zeros((self.num_paths, self.num_links), dtype=int)
        for i in range(self.num_paths):
            leaf_node = self.leaf_nodes[i]
            route_matrix[i][leaf_node - 1] = 1
            parent_node = self.tree_vector[leaf_node - 1]
            while parent_node != 0:
                route_matrix[i][parent_node - 1] = 1
                parent_node = self.tree_vector[parent_node - 1]
        return route_matrix

    def __get_depth(self):
        """
        获取树的深度，初始深度为0
        :return: depth:int
        """
        depth = 0
        for index in range(len(self.route_matrix)):
            depth = max(depth, sum(self.route_matrix[index]))
        return int(depth)

    def __str__(self):
        return "------------------cong_net--------------start"+"\n"+ \
               "树向量"+str(self.tree_vector)+"\n"+ \
                "所有的叶子节点"+str(self.leaf_nodes)+"\n"+ \
                "路径数量"+str(self.num_paths)+"\n"+\
                "链路数量"+str(self.num_links)+"\n"+\
                "路由矩阵"+"\n"+str(self.route_matrix)+"\n"+\
                "树的深度"+str(self.depth)+"\n"+\
                "链路拥塞相关属性:\n"+\
                "链路拥塞概率"+str(self.links_cong_pro)+"\n"+\
                "丢包率模型"+str(self.loss_model)+"\n"+\
                "丢包率门限"+str(self.threshold )+"\n"+\
                "各链路拥塞状态"+str(self.links_state_true)+"\n"+\
                "各链路丢包率"+str(self.links_loss_rate)+"\n"+\
                "真实拥塞链路编号列表"+str(self.links_cong_true)+"\n"+\
                "路径相关属性:"+"\n"+\
                "各路径丢包率"+str(self.paths_loss_rate)+"\n"+\
                "各路径真实拥塞状态"+str(self.paths_state_true)+"\n"+ \
                "真实拥塞路径编号列表" + str(self.paths_cong_true) + "\n" +\
                "真实非拥塞路径编号列表"+str(self.paths_no_cong_true)+"\n"+\
                "观测路径拥塞状态"+str(self.paths_state_obv)+"\n"+\
                "观测拥塞路径编号列表"+str(self.paths_cong_obv)+"\n"+\
                "观测非拥塞路径编号列表"+str(self.paths_no_cong_obv)+"\n"+ \
               "-------------------------cong_net_end------------------" + "\n"




def test_json_init():
    tree_vector=[0, 1, 1, 3, 3]
    file_path = os.path.join(os.path.dirname(os.getcwd()), "all_DS", str(tree_vector) + '.json')

    all_data={}
    with open(file_path,'r')as  fr:
        all_data=json.load(fr)

    basic_info=all_data["basic_info"]
    cong_data_info=all_data["[0, 0.1]"]["cong_data_info"][0]
    cong_config_info=all_data["[0, 0.1]"]["cong_config"]
    net=Cong_net()
    net.json_init(basic_info,cong_data_info,cong_config_info)
    print(net)

def test_cong_net():
    tree_vector = [0, 1, 1, 3, 3]
    links_congest_pro = utils.gen_link_congest_pro_1([0, 0.2], len(tree_vector))
    net = Cong_net()
    net.auto_init(tree_vector, "loss_model_1", 0.015,links_congest_pro)
    print(net)

    pass

if __name__ == '__main__':
    # test_cong_net()
    # test_save_ds_json()
    test_json_init()

    pass
