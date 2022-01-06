#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/24 16:25
# @Author  : Lelsey
# @File    : plot_net.py
import numpy as np
import matplotlib.pyplot as plt

def VTreetoE(VTree):
    E = []
    child = 1
    for parent in VTree:
        E.append((parent, child))
        child = child + 1
    return E
def getLeafNodes(VTree):
    '''
    :param VTree:
    :return: leafNodes
    '''
    leafNodes = []
    for i in range(len(VTree)):
        leafNode = i+1
        if leafNode not in VTree:
            leafNodes.append(leafNode)
    return leafNodes

def getChildren(E, parent):
    '''
    如果E不严格，那个输出的children顺序（左右）也不严格，用sort也没用
    :param E:
    :param parent:
    :return:
    '''
    children = []
    for edge in E:
        if edge[0] == parent:
            children.append(edge[1])
    return children

def to_setXi(node,x,E):
    parent = node
    children = getChildren(E, parent)
    sum = []
    for child in children:
        if x[child] == -1:
            to_setXi(child, x, E)
            sum.append(x[child])
        else:
            sum.append(x[child])
    temp = np.mean(sum)
    x[node] = np.mean(sum)


def TreePlot(VTree):
    '''
    nodes = np.array(VTree).reshape((1, len(VTree)))
    :param args:
    :return:
    '''
    E = VTreetoE(VTree)
    N = len(VTree)+1  ## 节点数目

    ## x的坐标 初始值为-1
    x = []
    for i in range(N):
        x.append(-1)
    x = np.array(x, dtype=float)
    ## 先设置x的坐标
    leafNodes = getLeafNodes(VTree)
    n = len(leafNodes)
    for i in range(n):
        x[i+1] = i
    ## 接着设置内节点
    for i in range(n+1, N):  # 等于孩子节点的坐标均值
        if x[i] == -1:
            parent = i
            children = getChildren(E, parent)
            sum = []
            for child in children:
                if x[child] == -1:
                    to_setXi(child,x,E)
                    sum.append(x[child])
                else:
                    sum.append(x[child])
            x[i] = np.mean(sum)
    ##设置根结点
    to_setXi(0,x,E)
    ##设置x的绝对位置：
    nn = n+2
    for i in range(N):
        x[i] = x[i]+1
    x_x = 1/nn
    for i in range(N):
        x[i] = x[i]*x_x

    ##设置y的坐标
    y = []
    for i in range(N):
        y.append(-1)
    y = np.array(y,dtype=float)
    layer = 0  ## 层数
    flag = True
    layer_visit = [0]
    while flag:
        next_layer = []
        while len(layer_visit) != 0:
            parent = layer_visit[0]
            y[parent] = layer
            del layer_visit[0]
            children = getChildren(E, parent)
            if len(children) != 0:
                next_layer.extend(children)
        layer = layer+1
        if len(next_layer) == 0:
            flag = False
        else:
            layer_visit = next_layer
    max_layer = np.max(y)
    ## 新把层数颠倒过来
    for i in range(N):
        y[i] = max_layer-y[i]
    # 预留空间 所以上下留一层
    max_layer = max_layer+2
    for i in range(N):
        y[i] = y[i]+1
    # 设置绝对位置
    y_y = 1/max_layer
    for i in range(N):
        y[i] = y[i]*y_y
    #画图
    plt.scatter(x,y,c='r',marker = 'o')
    for edge in E:
        parent = edge[0]
        child = edge[1]
        plt.plot([x[parent],x[child]],[y[parent],y[child]],color='b')
    for node in range(N):
        plt.annotate(text=str(node),xy=(x[node],y[node]),xytext=(x[node]+0.010,y[node]))
    plt.show()

if __name__ == '__main__':
    tree_vector=[0,1,1,3,3]
    TreePlot(tree_vector)
