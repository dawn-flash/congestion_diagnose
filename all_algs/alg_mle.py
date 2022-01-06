from scipy.optimize import minimize

import numpy as np


def alg_mle(y_loss, A_rm, initial:np.ndarray,num_package=1000):
    """
    mle算法的调用接口
    :param y_loss: 路径的丢包率，列向量；如为矩阵，横纬度为时间
    :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
    :param initial: array(n,1)  链路丢包率的初始值
    :param num_package: 每条路径上发送的探测报文个数
    :return: x_identified 为返回的已识别的链路拥塞状态；列向量，如为矩阵，横维度为时间，纵维度为链路推测状态
    """
    m,n=A_rm.shape
    _,num_times=y_loss.shape

    x_identified=np.zeros((n,num_times))
    for i in range(num_times):
        paths_loss_obv=y_loss[:,i].reshape(-1,1)
        # 生成路径报文接收与丢失情况
        paths_loss_obv = np.hstack((1000 * (1 - paths_loss_obv), num_times * paths_loss_obv)).round()
        links_state_infered=alg_mle_core(y_loss,A_rm,initial)[0].flatten()
        x_identified[:,i]=links_state_infered

    return x_identified

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

def log_likelihood(theta, y_loss, A_rm):
    # theta 为链路丢包率；
    # y_loss 为路径上观测的报文接收情况；纵维度指示不同的路径，横维度指示‘接收，丢包’的报文个数；
    # A_rm 为路由矩阵；纵维度为路径，横维度为链路；

    m, _ = A_rm.shape  # 获得路径的数目

    y_loss_pr = get_path_loss_pr(theta, A_rm)
    y_loss_loglike = 0.0
    for j in range(m):
        y_j_loss_loglike = y_loss[j, 0] * np.log(1-y_loss_pr[j]) + y_loss[j, 1] * np.log(y_loss_pr[j])

        y_loss_loglike += y_j_loss_loglike

    return y_loss_loglike

def alg_mle_core(y_loss, A_rm, initial:np.ndarray):
    """
    mle的核心代码
    :param y_loss: array(m,2) 路径的丢包情况数组 每一行：[接收数量，丢包数量]
    :param A_rm: array(m,n) 路由矩阵
    :param initial: array(n,1)  链路丢包率的初始值
    :return:(x_identified, x_lossrate_estimated)  array(n,1)推断链路的拥塞数组  array(n,1)推断的链路丢包率数组
    """
    _, n = A_rm.shape # 获取链路数目

    # 可以在最小化求解函数‘minimize’里，添加针对链路丢包率的bounds=bnds
    bnds = [(0.0, 1.0)] * n
    bnds = tuple(bnds)

    nll = lambda *args: -log_likelihood(*args)  # 定义目标函数

    soln = minimize(nll, initial, bounds=bnds, args=(y_loss, A_rm)) # 最大后验概率

    x_lossrate_estimated = soln.x.reshape((-1,1))
    x_identified = (x_lossrate_estimated > 0.01).astype(np.int8)
    return (x_identified, x_lossrate_estimated)


def test1():
    """
    随机产生数据进行测试
    :return:
    """
    # 设定随机种子
    np.random.seed(202106)

    # 路由矩阵，用于定义拓扑结构
    A_rm = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                     [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                     [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                     [1, 0, 1, 0, 1, 1, 0, 0, 0, 1]], np.int8)

    # A_rm = np.array([[1, 1, 0], [1, 0, 1]], np.int8)  # 倒'Y'型拓扑，2条路径，3条链路

    m, n = A_rm.shape  # 根据路由矩阵分别得到目标网络中路径与链路的数量

    num_times = 1000  # 每条路径上发送的探测报文个数

    x_pc = np.array([0.1] * n)  # 链路拥塞概率, 列向量

    x_true = np.zeros((n, 1), dtype=np.int8)  # 获取链路的真实拥塞状态
    x_true[0], x_true[-1] = 1, 1

    x_lossrate_true = np.random.rand(n, 1) * 0.01 + x_true * 0.01  # 获取链路的真实丢包率

    y_lossrate_true = get_path_loss_pr(x_lossrate_true, A_rm)  # 获取路径的真实丢包率

    y_loss = np.hstack((num_times * (1 - y_lossrate_true), num_times * y_lossrate_true)).round()  # 生成路径报文接收与丢失情况

    initial = np.random.rand(n, 1) * 0.001 + x_lossrate_true  # 初始化搜索位置，这里选择靠近真实解的位置；！！！很遗憾，发现初始化位置的不同会极大地影响求解性能

    x_identified, x_lossrate_estimated = alg_mle_core(y_loss, A_rm, initial)  # 识别出的链路拥塞状态

    print(f"x_identified:\n{x_identified}\nx_true:\n{x_true}\n")

# def test2():
#     """
#     读取数据进行测试
#     :return:
#     """
#     import os
#     from all_DS.config import Config
#     from all_tools.utils import get_drfpr_f1j
# # file_path=os.path.join(os.path.dirname(os.getcwd()),'all_DS','TOPO_DS','[0, 1, 1, 3, 3].json')
#     file_path=os.path.join(os.path.dirname(os.getcwd()),'all_DS','TOPO_DS','[0, 1, 1, 3, 3, 5, 5, 6, 6, 6].json')
#     print(file_path)
#     conf=Config(file_path)
#     links_pro_scope=conf.get_links_pro_scope()
#     A_rm=conf.get_routing_matrix()
#     m, n = A_rm.shape  # 根据路由矩阵分别得到目标网络中路径与链路的数量
#     paths_loss_rate=conf.get_paths_loss_rate(str(links_pro_scope[0]))
#     links_state_true=conf.get_links_state_true(str(links_pro_scope[0]))
#     route_matrix=conf.get_routing_matrix()
#
#     links_loss_true = conf.get_links_loss_rate("[0, 0.5]")[:,0].reshape(-1,1)
#
#     initial = np.random.rand(n, 1) * 0.001 + links_loss_true  # 初始化搜索位置，这里选择靠近真实解的位置；！！！很遗憾，发现初始化位置的不同会极大地影响求解性能
#     x_res=alg_mle(paths_loss_rate,route_matrix,initial)
#     print(x_res)
#
#     dr,fpr,f1,j=get_drfpr_f1j(links_state_true,x_res)
#     print("dr",dr)
#     print("fpr",fpr)
#     print("f1",f1)
#     print("j",j)


if __name__ == '__main__':
    # test2()
    test1()
    pass

