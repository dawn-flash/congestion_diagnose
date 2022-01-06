#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 15:41
# @Author  : Lelsey
# @File    : clink_xt.py

import numpy as np
import random
from all_algs.base_alg import BaseAlg
from all_nets.cong_net import Cong_net
from all_tools import utils
import copy






#------------------------------------clink算法类-----------------------
class CLINK(BaseAlg):  # 继承 alg_SCFS 父类
    def __init__(self, net:Cong_net):
        super(CLINK, self).__init__(net)

    def __get_link_state_class(self, un_congested_path:list):
        """
        根据非拥塞路径，返回正常链路列表，和拥塞链路列表
        :param un_congested_path:list
        :return:good_link:list ,uncertain_link:list   存储链路下标
        """
        # 所有经过了不拥塞路径的链路
        good_link = []
        for i in un_congested_path:
            for index, item in enumerate(self.route_matrix[i]):
                if int(item) == 1 and index not in good_link:
                        good_link.append(index)

        all_links = [i for i in range(self.num_links)]
        # 排除那些肯定不拥塞的链路
        uncertain_link = []
        for item in all_links:
            if item not in good_link:
                uncertain_link.append(item)
        return good_link, uncertain_link

    def diagnose(self):
        print('链路的拥塞概率:', self.links_cong_pro)
        congested_path = (self.paths_cong_obv - 1).tolist()
        un_congested_path = (self.paths_no_cong_obv - 1).tolist()
        print("congested_path",congested_path)
        print("un_congested_path",un_congested_path)

        #生成正常链路和不确定链路
        good_link, uncertain_link = self.__get_link_state_class(un_congested_path)
        print('位于不拥塞路径中的链路:', good_link)
        print('不确定拥塞状态的链路:', uncertain_link)

        #获取经过一条链路的所有路径domain
        domain_dict = {}
        for i in uncertain_link:
            domain_dict[i] = [j for j in self.net.get_paths(i+1) if j in congested_path]
        print("domain_dict")
        print(domain_dict)

        self.links_state_inferred = np.zeros(self.num_links)

        # 计算所有的链路
        temp_state = [1e8 for _ in range(len(uncertain_link))]
        print('temp_state:', temp_state)
        while len(congested_path) > 0:
            # 找到最小的值对应的链路
            for index, i in enumerate(uncertain_link):
                # print(self._congestion_prob_links)
                #方法1 公式(log((1-p)/p))|domain(x)|
                a = np.log((1 - self.links_cong_pro[i]) / (self.links_cong_pro[i]))
                b = len(domain_dict[i])
                if b==0:
                    temp_state[index]=1e8
                else:
                    temp_state[index] = a / b

                #方法2 公式log((1-p)/p/|domain(x)|)
                # b=len(domain_dict[i])
                # if b==0:
                #     temp_state[index]=1e8
                # else:
                #     a=np.log((1 - self._congestion_prob_links[i]) / (self._congestion_prob_links[i])/b)
                #     temp_state[index]=a

            print(temp_state)
            index = temp_state.index(min(temp_state))
            self.links_state_inferred[uncertain_link[index]] = 1
            self.links_cong_inferred.append(uncertain_link[index] + 1)
            print("推断的链路",uncertain_link[index]+1)
            for item in domain_dict[uncertain_link[index]]:
                if item in congested_path:
                    print('congested_path', congested_path)
                    print('item:', item)
                    congested_path.remove(item)
            domain_dict.pop(uncertain_link[index])
            uncertain_link.remove(uncertain_link[index])
            temp_state.remove(temp_state[index])

            for k,v in domain_dict.items():
                temp=[]
                for i in v:
                    if i in congested_path:
                        temp.append(i)
                domain_dict[k]=copy.deepcopy(temp)


            print("domain_dict")
            print(domain_dict)
            print("uncertain_link",uncertain_link)
            print("congest_path",congested_path)

        print("真实的链路拥塞",self.links_cong_true)
        print('推测的链路拥塞:', self.link_state_inferred)

    @property
    def link_state_inferred(self):
        return self.links_state_inferred

    def __str__(self):
        """
        tree_vector:list
        loss_modelstr
        _congestion_prob_links:list
        state_paths_real:ndarray
        _state_links:ndarray
        _links_congested:ndarray
        _paths_congested:ndarray
        _congested_link_inferred:list
        _link_state_inferred:ndarray
        _dr:float
        fpr:float
        :return:
        """
        return "clink-------------------------------------------start\n"+\
                "网络相关属性\n"+\
                "tree_vector"+str(self.net.tree_vector)+'\n'+ \
                "loss_model" + str(self.net.loss_model) + '\n' + \
                "links_cong_pro" + str(self.links_cong_pro) + '\n' + \
                "路径相关属性\n"+\
                "paths_state_true"+str(self.paths_state_true)+'\n'+ \
                "paths_cong_true" + str(self.paths_cong_true) + '\n' + \
                "paths_state_obv"+str(self.paths_state_obv)+'\n'+\
                "paths_cong_obv"+str(self.paths_cong_obv)+'\n'+\
                "链路相关属性\n"+\
                "links_state_true" + str(self.links_state_true) + '\n' + \
                "links_cong_true" + str(self.links_cong_true) + '\n'+ \
                "推测结果和相关属性\n"+\
                "links_state_inferred" + str(self.links_state_inferred) + '\n' + \
                "links_cong_inferred" + str(self.links_cong_inferred) + '\n' + \
                "_dr" + str(self._dr) + '\n' + \
                "fpr" + str(self._fpr) + '\n'+\
                "f1_score"+str(self.f1_score)+'\n'+ \
                "clink-------------------------------------------end\n"

def test_clink1():
    random.seed(0)
    np.random.seed(0)
    tree_vector = [0, 1, 1, 3, 3]
    links_congest_pro = utils.gen_link_congest_pro_1([0, 0.5], len(tree_vector))
    net = Cong_net()
    net.auto_init(tree_vector, "loss_model_1", 0.015, links_congest_pro)
    print(net)

    clink=CLINK(net)
    clink.diagnose()
    clink.evaluation()
    print(clink)
    pass

def test_clink2():
    tree_vector = [0, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 9, 9, 12, 12, 15, 15]
    loos_model = "loss_model_1"
    link_cong_prog = [0.549, 0.715, 0.603, 0.545, 0.424, 0.646, 0.438, 0.892, 0.964, 0.383, 0.792, 0.529, 0.568, 0.926,
                      0.071, 0.087, 0.02, 0.833, 0.778]
    # link_cong_prog=0.1
    threshold = 0.01
    net=Cong_net()
    net.auto_init(tree_vector,loos_model,threshold,link_cong_prog)

    clink=CLINK(net)

    clink.paths_state_true=np.array([0 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1])
    clink.paths_cong_true=np.array([ 2 , 3 , 5 , 6 , 7 , 8 , 9 ,10 ,11 ,12 ,13 ,14])
    clink.paths_no_cong_true=np.array([1,4])

    clink.paths_state_obv=np.array([0 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1])
    clink.paths_cong_obv=np.array([ 2 , 3 , 5 , 6 , 7 , 8 , 9 ,10 ,11 ,12 ,13 ,14])
    clink.paths_no_cong_obv=np.array([1,4])

    clink.links_state_true=np.array([0 ,0 ,1 ,1 ,0 ,1 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,0 ,0 ,0 ,0])
    clink.links_cong_true=np.array([ 3 ,4 , 6 , 8 , 9 ,12 ,13 ,14])
    clink.diagnose()
    clink.evaluation()
    print(clink)

def test_clink3():
    tree_vector = [0, 1, 1, 3, 3, 5, 5, 6, 6, 6]
    loos_model = "loss_model_1"
    link_cong_prog = 0.1
    # link_cong_prog=0.1
    threshold = 0.01
    net=Cong_net()
    net.auto_init(tree_vector,loos_model,threshold,link_cong_prog)
    print(net)
    clink=CLINK(net)

    clink.paths_state_true=np.array([1,0,1,1,1,1])
    clink.paths_cong_true=np.array([1,3,4,5,6])
    clink.paths_no_cong_true=np.array([2])

    clink.paths_state_obv=np.array([1,0,1,1,1,1])
    clink.paths_cong_obv=np.array([1,3,4,5,6])
    clink.paths_no_cong_obv=np.array([2])

    clink.links_state_true=np.array([0,1,0,0,1,0,0,0,0,0])
    clink.links_cong_true=np.array([2,5])
    clink.diagnose()
    clink.evaluation()
    print(clink)


if __name__ == '__main__':
    # test_clink1()
    # test_clink2()
    # test_clink3()

    pass