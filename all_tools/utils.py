#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 9:51
# @Author  : Lelsey
# @File    : utils.py
import numpy as np

def gen_link_congest_pro_1(pro_score:list,link_nums:int):
    """
    根据均匀分布随机为链路分配拥塞概率
    拥塞概率取值pro_score
    :param pro_score: 拥塞概率范围
    :param link_nums: 链路数量
    :return:
    """
    link_congest_list=[]
    for i in range(link_nums):
        while True:
            temp=np.random.uniform(pro_score[0],pro_score[1])
            temp=round(temp,3)
            if (temp not in link_congest_list)and temp!=0:
                link_congest_list.append(temp)
                break
    # link_congest_list=np.random.uniform(0,pro_score,link_nums).tolist()
    return link_congest_list

def cal_f1_score_xt(link_congested_true:list,link_congested_infer:list):
    """
    计算f1_score
    :param link_congested_true: 真实链路拥塞列表
    :param link_congested_infer: 推测链路拥塞列表
    :return: f1_score：float
    """

    TP = len(set(link_congested_infer).intersection(set(link_congested_true)))
    FN = len(set(link_congested_true) - (set(link_congested_infer).intersection(set(link_congested_true))))
    FP = len(set(link_congested_infer) - (set(link_congested_infer).intersection(set(link_congested_true))))

    f1_score=0
    precison = 0
    recall = 0
    if len(link_congested_true) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if len(link_congested_infer) == 0:
        precison = 0
    else:
        precison = TP / (TP + FP)
    if precison + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * recall * precison) / (recall + precison)

    return f1_score

def test_cal_f1_score():
    true_list=[3,4,6,8,9,12,13,14]
    infer_list=[14, 8, 9, 11, 3, 13, 4, 6]
    res=cal_f1_score_xt(true_list,infer_list)
    print(res)

def cal_eachlink_dr_fpr(link_tp_fp_fn_tn_list:list,num_links):
    """
    重复n次实验，计算出n次实验的tpfpfntn，然后计算每一条链路的dr和fpr
    :param link_tp_fp_fn_tn_list:
    :param num_links:
    :return:
    """
    all_res=np.zeros(shape=(num_links,4))

    for i in range(len(link_tp_fp_fn_tn_list)):
        for j in range(len(link_tp_fp_fn_tn_list[i])):
            link_type=link_tp_fp_fn_tn_list[i][j]
            all_res[j][link_type-1]+=1

    dr_list=[]
    fpr_list=[]
    for i in range(all_res.shape[0]):
        TP=all_res[i][0]
        FP=all_res[i][1]
        FN=all_res[i][2]
        TN=all_res[i][3]
        dr=0
        fpr=0
        if TP+FN==0:
            dr=0
        else:
            dr=TP/(TP+FN)
        if TP + FP == 0:
            fpr=0
        else:
            fpr=FP / (TP + FP)
        dr_list.append(dr)
        fpr_list.append(fpr)
    return dr_list,fpr_list,all_res

def cal_tp_fp_fn_tn(link_congested_true: list, link_congested_infer: list, link_num: int):
    link_state_true = [0] * link_num
    link_state_infer = [0] * link_num
    for i in link_congested_true:
        link_state_true[i - 1] = 1
    for i in link_congested_infer:
        link_state_infer[i - 1] = 1

    res_list = []
    for i in range(link_num):
        if link_state_true[i] == 0 and link_state_infer[i] == 0:
            tn = 4
            res_list.append(tn)
        if link_state_true[i] == 0 and link_state_infer[i] == 1:
            fp = 2
            res_list.append(fp)
        if link_state_true[i] == 1 and link_state_infer[i] == 0:
            fn = 3
            res_list.append(fn)
        if link_state_true[i] == 1 and link_state_infer[i] == 1:
            tp = 1
            res_list.append(tp)
    return res_list

def test_link_dr_fpr():
    link_congested_true=[1,2,5,6,7]
    link_congested_infer=[2,6,7,8,10]
    link_num=10

    res=[]
    res=cal_tp_fp_fn_tn(link_congested_true,link_congested_infer,link_num)
    print("结果为")
    print(res)



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

#-----------------------------------------老师给的工具包--------------------------------------------------------
def get_drfpr_f1j(x_true, x_identified, target_label = 1, output = 'averaged', eachlink = False):
    # x_true 链路真实的拥塞状态，列向量；如为矩阵，横维度为时间；
    # x_identified 已识别链路拥塞状态，列向量；如为矩阵，横维度为时间；
    # target_label 指示拥塞状态的值；默认 1 表示拥塞状态，0 为正常状态；
    # output 指示输出度量值的形式；默认为平均值‘averaged’；‘all’为输出每个时刻的度量值
    #eachlink 指示是否为针对每条链路来计算其拥塞状态识别性能的度量值；默认统计全部链路；

    if eachlink: # 针对每条链路来计算其各自的度量值
       output = 'all'  # 此时必须指定输出全部链路的度量值

       x_true = x_true.transpose()
       x_identified = x_identified.transpose()

    target_hit = (x_true == target_label).astype(np.int32) # 标定出每个时刻真实发生拥塞的链路
    target_num = np.sum(target_hit, axis=0) # 每个时刻真实发生拥塞的链路数量

    num_links = np.size(x_true, axis=0) # 网络中链路的数量
    num_times = np.size(target_num) # 时刻数 或重复实验的总次数

    DR = np.ones(num_times, np.float64)
    FPR = np.zeros(num_times, np.float64)
    F1 = np.ones(num_times, np.float64)
    J = np.ones(num_times, np.float64)
    F2=np.ones(num_times,np.float64)

    for t in range(num_times):
        if target_num[t] == 0: # 如果没有链路发生拥塞
            continue
        else:
            x_id = x_identified[:, t]
            x_tr = x_true[:, t]

            index = np.where(x_id == target_label)
            hit_index = (x_id[index] == x_tr[index]).astype(np.int32)

            tp = np.sum(hit_index)
            fn = target_num[t] - tp
            fp = np.size(x_id[index]) - tp

            DR[t] = tp / target_num[t]

            if target_num[t] == num_links: # 如果所有链路全部发生拥塞
                FPR[t] = 0.0
            else:
                FPR[t] = fp / (num_links - target_num[t])

            F1[t] = 2 * tp / (2 * tp + fn + fp)
            J[t] = tp / (tp + fp + fn)

            #计算F2的值

            F2[t]=(5*tp)/(5*tp+fp+fn)


    # 根据给定的输出形式，返回度量值
    if output == 'averaged':  # 默认输出所有时刻的平均度量值
        return (DR.mean(), FPR.mean(), F1.mean(), J.mean(),F2.mean())
    elif output == 'all':     # 输出每个时刻的度量值
        return (DR, FPR, F1, J,F2)

    # 这个else无法执行到
    # else:
    #     print('注意：返回的是每个时刻的度量值。')
    #     return (DR, FPR, F1, J)

def get_path_loss_pr(theta, A_rm):
    # theta 为链路丢包率；
    # A_rm 为路由矩阵；纵维度为路径，横维度为链路；

    m, _ = A_rm.shape  # 获得路径的数目

    theta_trans = 1.0 - theta # 得到链路的传输率
    y_loss_pr = np.zeros((m, 1))
    for j in range(m):
        route = np.where(A_rm[j, :] >0) # 标记路径经过的链路
        y_loss_pr[j] = 1.0 - np.prod(theta_trans[route]) # 得到路径的丢包率

    return y_loss_pr

def test_get_drfpr_f1j():
    # x_true=np.array([1,0,1,0,1]).reshape((-1,1))
    # x_identified=np.array([1,1,0,1,0]).reshape((-1,1))

    # x_true=np.array([1,1,0,1,0]).reshape((-1,1))
    # x_identified=np.array([1,0,1,0,1]).reshape((-1,1))

    # x_true=np.array([1,1,0,0,1]).reshape((-1,1))
    # x_identified=np.array([1,0,1,0,1]).reshape((-1,1))

    # x_true=np.array([[1,0,1,0,1],[1,1,0,0,1]]).transpose()
    # x_identified = np.array([[1, 1, 0, 1, 0],[1,0,1,0,1]]).transpose()
    x_true=np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    x_identified=np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    target_label=1
    output="averaged"
    eachlink=False
    all_res=get_drfpr_f1j(x_true,x_identified,target_label,output,eachlink)
    print(all_res)


if __name__ == '__main__':
    test_get_drfpr_f1j()