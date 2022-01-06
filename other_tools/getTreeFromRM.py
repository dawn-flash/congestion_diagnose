#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 9:54
# @Author  : Lelsey
# @File    : getTreeFromRM.py

import numpy as np
class getTreeFromRm:
    def __init__(self):
        pass

    def gen_Tree_from_RM(self, routing_matrix):  # 依据路由矩阵来得到树型拓扑, 并没有检查路由矩阵的有效性
        """
        将路由矩阵转换为树向量
        :param routing_matrix: np.ndarray(m,n)
        :return: self.linkSetVec
        """
        self.routing_matrix=routing_matrix
        #将路由矩阵转换为按列排序后的路由矩阵
        routing_matrix_sorted,y_list_sorted = self.rm_to_rm_sorted(routing_matrix)
        self.path_num, self.link_num = np.shape(routing_matrix)
        node_dist = np.arange(self.link_num + 1)
        tree_vector = [0] * (self.link_num)
        i_node = []
        for i in range(self.path_num):
            path = routing_matrix_sorted[i]
            pre_node = 0
            for j in range(path.size):
                if path[j] == 1:
                    tree_vector[j] = pre_node
                    i_node.append(pre_node)
                    pre_node = j + 1

        self.dist = np.delete(node_dist, i_node)
        self.linkSetVec = np.asarray([tree_vector, range(1, self.link_num + 1)]).transpose()
        restore_index = np.argsort(y_list_sorted)
        self.linkSetVec = self.linkSetVec[restore_index, :]


    def rm_to_rm_sorted(self,A_rm: np.ndarray):
        """
        将路由矩阵转换为排序后的矩阵
        :param A_rm: np.ndarray(m,n)
        :return:  A_rm_sorted(m,n) 排序后的矩阵, y_list_sorted 排序后的列编号
        """
        y_list = self.column_insert_sort(A_rm)
        A_rm_sorted = np.zeros(A_rm.shape)
        for i in range(len(y_list)):
            A_rm_sorted[:, i] = A_rm[:, y_list[i]]
        return A_rm_sorted,y_list

    def column_cmp(self,y1, y2, A_rm):
        """
        列排序函数
        :param y1: y1列
        :param y2: y2列
        :param A_rm: np.ndarray(m,n)路由矩阵
        :return:
        """
        m = A_rm.shape[0]
        y1_set = set([i for i in range(m) if A_rm[i, y1] > 0])
        y2_set = set([i for i in range(m) if A_rm[i, y2] > 0])
        if y1_set.issubset(y2_set):
            return True
        return False

    def column_insert_sort(self,A_rm):
        """
        对矩阵的列进行排序
        :param A_rm:
        :return:
        """
        n = A_rm.shape[1]
        y_list = list(range(0, n))
        for i in range(1,n):
            inser_index=i
            inser_num=y_list[inser_index]
            for j in range(i-1,-1,-1):
                if self.column_cmp(y_list[j], y_list[i], A_rm):
                    inser_index=j
            for k in range(i,inser_index,-1):
                y_list[k]=y_list[k-1]
            y_list[inser_index]=inser_num
        print("新列的排序后编号")
        print(y_list)
        return y_list



def test1():
    """
    正常矩阵:[[1,1,0,0,0],
            [1,0,1,1,0],
            [1,0,1,0,1]]
            逆序测试
    :return:
    """
    get_tree=getTreeFromRm()
    RM=np.array([[0,0,0,1,1],
                 [0,1,1,0,1],
                 [1,0,1,0,1]])
    get_tree.gen_Tree_from_RM(RM)
    print(get_tree.linkSetVec)

def test2():
    get_tree=getTreeFromRm()
    RM=np.array([[0,0,0,1,1],
                 [1,0,1,0,1],
                 [0,1,1,0,1]])
    get_tree.gen_Tree_from_RM(RM)
    print(get_tree.linkSetVec)

def test3():
    get_tree=getTreeFromRm()
    RM=np.array([[0,0,0,1,1],
                 [1,1,0,0,1],
                 [1,0,1,0,1]])
    get_tree.gen_Tree_from_RM(RM)
    print(get_tree.linkSetVec)

def test3():
    get_tree=getTreeFromRm()
    RM=np.array([[0,0,0,1,1],
                 [1,1,0,0,1],
                 [1,0,1,0,1]])
    get_tree.gen_Tree_from_RM(RM)
    print(get_tree.linkSetVec)

def test4():
    get_tree=getTreeFromRm()
    tree_vector=[0,1,1,3,3,5,5,6,6,6]
    RM=np.array([
                [1,1,0,0,0,0,0,0,0,0],
                [1,0,1,1,0,0,0,0,0,0],
                [1,0,1,0,1,0,1,0,0,0],
                [1,0,1,0,1,1,0,1,0,0],
                [1,0,1,0,1,1,0,0,1,0],
                [1,0,1,0,1,1,0,0,0,1]])
    get_tree.gen_Tree_from_RM(RM)
    print("链路列表")
    print(get_tree.linkSetVec)

def test5():
    """
    逆序测试
    :return:
    """
    get_tree=getTreeFromRm()
    tree_vector=[0,1,1,3,3,5,5,6,6,6]
    RM = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 1, 0, 1]])
    get_tree.gen_Tree_from_RM(RM)
    # get_tree.column_insert_sort(RM)
    print("变换后链路列表")
    print(get_tree.linkSetVec)

def test6():
    """
    正常矩阵:[[1,1,0,0,0],
            [1,0,1,1,0],
            [1,0,1,0,1]]
            链路1后移到链路5
    :return:
    """
    get_tree=getTreeFromRm()
    RM=np.array([[1, 0, 0, 0, 1],
                 [0, 1, 1, 0, 1],
                 [0, 1, 0, 1, 1]])
    get_tree.gen_Tree_from_RM(RM)
    print(get_tree.linkSetVec)

if __name__ == '__main__':
    # print("test1-------------------------")
    # test1()
    # print("test2-------------------------")
    # test2()
    # print("test3------------------------")
    # test3()
    # print("test4----------------------------")
    # test4()
    # print("test5----------------------")
    test5()
    # print("test6------------------------")
    # test6()