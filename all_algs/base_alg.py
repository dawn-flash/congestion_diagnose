#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 16:44
# @Author  : Lelsey
# @File    : base.py

import copy
import numpy as np
from all_nets.cong_net import Cong_net
from all_tools import utils

class BaseAlg:
    def __init__(self, net:Cong_net):
        """
        所有算法的基类，
        diagnose()  该函数进行拥塞链路的诊断
        evaluation() 该函数用于诊断结果的相关指标计算

        算法生成的推测信息：
        self.links_cong_inferred：list  # 诊断为拥塞的链路编号列表
        self.links_state_inferred：ndarray  # 诊断的链路拥塞的状态
        self.dr:float
        self.fpr:float
        self.f1_score:float
        :param net:  拥塞网络类
        """
        self.net = net

        # topo基本信息
        self.leaf_nodes = net.leaf_nodes  #叶子结点数组
        self.route_matrix = net.route_matrix  #路由矩阵
        self.num_links=net.num_links   #链路数量

        # 链路拥塞相关属性
        self.links_cong_pro = net.links_cong_pro  #链路失效概率
        self.loss_model = net.loss_model   #丢包率模型
        self.threshold = net.threshold  #丢包率门限

        # 链路相关信息
        self.links_state_true = net.links_state_true    #真实链路状态数组
        self.links_loss_rate = net.links_loss_rate      #真实链路丢包率数组
        self.links_cong_true = net.links_cong_true      #真实链路拥塞数组

        # 真实路径相关属性
        self.paths_loss_rate = net.paths_loss_rate    #真实路径丢包率数组
        self.paths_state_true = net.paths_state_true  #真实路径状态数组
        self.paths_cong_true =net.paths_cong_true     #真实拥塞路径数组
        self.paths_no_cong_true=net.paths_no_cong_true  #真实非拥塞路径数组

        # 观测的路径相关属性
        # self.paths_state_obv =copy.deepcopy(net.paths_state_obv )   #观测路径状态数组
        # self.paths_cong_obv=copy.deepcopy(net.paths_cong_obv)       #观测拥塞路径数组
        # self.paths_no_cong_obv=net.paths_no_cong_obv  #观测非拥塞路径数组
        #注意，目前将观测的路径状态设置为真实的路径状态
        self.paths_state_obv = copy.deepcopy(net.paths_state_true)  # 观测路径状态数组
        self.paths_cong_obv = copy.deepcopy(net.paths_cong_true)  # 观测拥塞路径数组
        self.paths_no_cong_obv = net.paths_no_cong_true  # 观测非拥塞路径数组

        #推测结果信息
        self.links_cong_inferred = []  # 诊断为拥塞的链路列表
        self.links_state_inferred = None  # 诊断的链路拥塞的状态
        self.dr = None
        self.fpr = None
        self.f1_score=None

    def diagnose(self):
        """
        拥塞链路诊断，由子类继承
        :return:
        """
        pass

    def evaluation(self):
        """
        拥塞链路评估方法1
        dr和fpr的计算方式1
        dr=(推测拥塞 & 真实拥塞)/(真实拥塞数量)
        fpr=(推测拥塞-真实拥塞)/(真实不拥塞数量)
        :return:
        """
        if np.any(self.net.links_state_true):  # 判断是否不存在拥塞链路

            self._dr = len(set(self.links_cong_inferred).intersection(set(self.links_cong_true))) \
                      / len(self.links_cong_true)
        else:
            self._dr = 1.0

        if len(self.links_cong_true) < self.num_links:  # 判断是否不存在正常链路
            self._fpr = len(set(self.links_cong_inferred).difference(set(self.links_cong_true))) \
                       / (self.num_links - len(self.links_cong_true))
        else:
            self._fpr = 0.0

        #计算f1_score
        self.f1_score=utils.cal_f1_score_xt(list(self.links_cong_true),list(self.links_cong_inferred))

        #计算链路的dr和fpr

        #统计链路的tp，fp，fn，tn

    def evaluation_1(self):
        """
        拥塞链路评估方法2
        dr和fpr的计算方式2
        dr=(推测拥塞 & 真实拥塞)/(真实拥塞数量)
        fpr=(推测拥塞-真实拥塞)/(推测拥塞的数量)
        :return:
        """
        if np.any(self.links_state_true):  # 判断是否不存在拥塞链路

            self._dr = len(set(self.links_cong_inferred).intersection(set(self.links_cong_true))) \
                      / len(self.links_cong_true)
        else:
            self._dr = 1.0

        # if len(self._links_congested) < self._num_links:  # 判断是否不存在正常链路
        if len(self.links_cong_inferred)!=0:
            self._fpr = len(set(self.links_cong_inferred).difference(set(self.links_cong_true))) \
                       / (len(set(self.links_cong_inferred)))
        else:
            self._fpr = 0.0

        self._f1_score = utils.cal_f1_score_xt(list(self.links_cong_true), list(self.links_cong_inferred))

