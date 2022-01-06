#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/6 20:11
# @Author  : Lelsey
# @File    : plot_experiment.py
import matplotlib.pyplot as plt
import os
import json


def plot_data(data,alg_list,x_list,x_label,y_label,title_name):

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure()
    scfs_y=data[alg_list[0]]
    clink_y=data[alg_list[1]]
    map_y=data[alg_list[2]]

    plt.plot(x_list, scfs_y, 'ro-', label=alg_list[0])
    plt.plot(x_list, clink_y, 'cs-', label=alg_list[1])
    plt.plot(x_list, map_y, 'bv-', label=alg_list[2])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.xticks(x_list)
    plt.title(u""+title_name)  # u加上中文
    plt.show()

def plot_Bellcanada_isomorphism():
    """
    画出同构下曲线Bellcanada_isomorphism.json中的数据
    /experiment_res/motivation_res/Bellcanada_isomorphism.json
    :return:
    """
    all_data = {}
    filepath = os.path.join(os.getcwd(), "experiment_res", "motivation_res", "Bellcanada_isomorphism.json")
    with open(filepath, 'r')as fr:
        all_data = json.load(fr)

    #画dr图
    evaluate_data = all_data["evaluate_data"]["dr"]
    alg_list = ["scfs", "clink", "map"]
    x_list = all_data["error_list"]
    x_label = "error"
    y_label = "dr"
    title_name = ""
    plot_data(evaluate_data, alg_list, x_list, x_label, y_label, title_name)

    #画fpr图
    evaluate_data = all_data["evaluate_data"]["fpr"]
    alg_list = ["scfs", "clink", "map"]
    x_list = all_data["error_list"]
    x_label = "error"
    y_label = "fpr"
    title_name = ""
    plot_data(evaluate_data, alg_list, x_list, x_label, y_label, title_name)

    #画f1图
    evaluate_data = all_data["evaluate_data"]["f1"]
    alg_list = ["scfs", "clink", "map"]
    x_list = all_data["error_list"]
    x_label = "error"
    y_label = "f1"
    title_name = ""
    plot_data(evaluate_data, alg_list, x_list, x_label, y_label, title_name)

    # 画f2图
    evaluate_data = all_data["evaluate_data"]["f2"]
    alg_list = ["scfs", "clink", "map"]
    x_list = all_data["error_list"]
    x_label = "error"
    y_label = "f2"
    title_name = ""
    plot_data(evaluate_data, alg_list, x_list, x_label, y_label, title_name)




#-------------------------------测试代码--------------------------------------
def test():
    all_data={}
    filepath=os.path.join(os.getcwd(),"experiment_res","test","first_isomorphism_test.json")
    with open(filepath,'r')as fr:
        all_data=json.load(fr)

    evaluate_data=all_data["evaluate_data"]["dr"]
    alg_list=["scfs","clink","map"]
    x_list=all_data["error_list"]
    x_label="error"
    y_label="dr"
    title_name=""
    plot_data(evaluate_data,alg_list,x_list,x_label,y_label,title_name)


if __name__ == '__main__':
    # test()
    plot_Bellcanada_isomorphism()