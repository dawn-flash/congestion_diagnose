#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 15:42
# @Author  : Lelsey
# @File    : scfs_xt.py


import numpy as np
from all_algs.base_alg import BaseAlg
from all_nets.cong_net import Cong_net
from all_tools import utils
import copy
import random

class SCFS(BaseAlg):
    def __init__(self, net:Cong_net):  # 类初始化
        super(SCFS, self).__init__(net)

    def diagnose(self):
        self.links_state_inferred = np.zeros(self.num_links + 1)  # 链路初始诊断为正常状态
        self.paths_state_obv_temp=copy.deepcopy(self.paths_state_obv) #设置一个临时变量
        self.algorithm(1)  # 从链路 l_1 开始诊断

        self.links_cong_inferred.sort()
        self.links_state_inferred = self.links_state_inferred[1:]  # 去掉根节点0所在的虚拟链路状态

    def algorithm(self, k):
        if k not in self.leaf_nodes:
            d = self.net.get_children(k)

            path, = np.where(self.route_matrix[:, k - 1] > 0)
            self.links_state_inferred[k] = np.min(self.paths_state_obv_temp[path])

            if self.links_state_inferred[k]:
                self.links_state_inferred[k] = 1  # 强制设为布尔状态
                self.paths_state_obv_temp[path] = 0  # 将路径状态重置为0

            for i in d:
                self.algorithm(i)  # 递归操作

        else:
            path = self.net.get_paths(k)
            self.links_state_inferred[k] = self.paths_state_obv_temp[path]

            if self.links_state_inferred[k]:
                self.links_state_inferred[k] = 1  # 强制设为布尔状态

            self.paths_state_obv_temp[path] = 0  # 将所有路径状态重置为0

        if self.links_state_inferred[k]:
            self.links_cong_inferred.append(k)

    @property
    def link_state_inferred(self):
        return self.links_state_inferred

    @property
    def congested_link_inferred(self):
        return self.links_cong_inferred

    def __str__(self):
        return "scfs-------------------------------------------start\n"+\
                "网络相关属性\n"+\
                "tree_vector"+str(self.net.tree_vector)+'\n'+ \
                "loss_model" + str(self.loss_model) + '\n' + \
                "links_cong_pro" + str(self.links_cong_pro) + '\n' + \
                "路径相关属性"+'\n'+\
                "paths_state_true"+str(self.paths_state_true)+'\n'+ \
                "paths_cong_true"+str(self.paths_cong_true)+'\n'+\
                "paths_state_obv"+str(self.paths_state_obv)+'\n'+\
                "paths_cong_obv"+str(self.paths_cong_obv)+'\n'+\
                "链路相关属性"+'\n'+\
                "links_state_true" + str(self.links_state_true) + '\n' + \
                "links_cong_true" + str(self.links_cong_true) + '\n'+ \
                "推测结果和相关指标\n"+\
                "links_state_inferred" + str(self.links_state_inferred) + '\n' + \
                "links_cong_inferred" + str(self.links_cong_inferred) + '\n' + \
                "_dr" + str(self._dr) + '\n' + \
                "fpr" + str(self._fpr) + '\n'+ \
                "f1_score"+str(self.f1_score)+'\n'+\
                "scfs-----------------------------------------end\n"




#测试函数---------------------------------------------------------------
def test_scfs():
    """
    scfs算法的测试例子
    :return:
    """
    # random.seed(2)
    # np.random.seed(2)
    tree_vector = [0, 1, 1, 3, 3]
    links_congest_pro = utils.gen_link_congest_pro_1([0, 0.5], len(tree_vector))
    net = Cong_net()
    net.auto_init(tree_vector, "loss_model_1", 0.015, links_congest_pro)
    print(net)

    scfs=SCFS(net)
    scfs.diagnose()
    scfs.evaluation()
    print(scfs)

if __name__ == '__main__':
    test_scfs()
    pass
