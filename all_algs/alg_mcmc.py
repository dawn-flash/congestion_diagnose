import zeus
''' [INFOCOM'03] Server-based Inference of Internet Link Lossiness '''

import numpy as np
from multiprocessing import Pool

def logprior(theta_x_lossrate):
    # 假设均匀先验；或者假设不具备链路丢包率的任何先验知识

    lp = 1.0
    return np.log(lp)

def get_path_likelihood(j, l_L, A_rm):
    # j 为第‘j’条路径；
    # l_L 为链路丢包率的向量；
    # A_rm 为路由矩阵；横维度为时间；

    _, n = A_rm.shape  # 获取链路数目

    pr_path = 1.0
    for i in range(n):
        if A_rm[j, i]:
            pr_path = pr_path * (1.0 - l_L[i]) # 一条路径的传输率为其上各条链路传输率的乘积

    pr_path = 1.0 - pr_path # 得到该条路径的路径丢包率
    return pr_path

def loglike(theta_x_lossrate, y, A_rm):
    # 根据拓扑来计算对数似然

    m, n = A_rm.shape  # 分别获取目标网络的路径与链路的总数量

    pr_post = 0.0
    for j in range(m):
        pr_path = get_path_likelihood(j, theta_x_lossrate, A_rm)

        pr_post = pr_post + y[j, 0] * np.log(1 - pr_path) + y[j, 1] * np.log(pr_path)

    # return the log likelihood
    return pr_post

def logpost(theta_x_lossrate, y, A_rm):
    '''The natural logarithm of the posterior.'''

    return logprior(theta_x_lossrate) + loglike(theta_x_lossrate, y, A_rm)

def alg_mcmc(y, A_rm):

    _, n = A_rm.shape  # 分别获取目标网络的路径与链路的总数量

    ndim = n # 未知参数个数；这里为链路数
    nwalkers = 2 * ndim # 模拟的chain数目
    nsteps = 2000 # 迭代次数

    start = 0.01 + 0.01 * np.random.rand(nwalkers, ndim) # 初始化各条chain的位置

    with Pool() as pool: # 开启并行，提升计算模拟速度
        sampler = zeus.EnsembleSampler(nwalkers, ndim, logpost, args=[ y, A_rm], pool=pool) # 初始化mcmc模拟器
        sampler.run_mcmc(start, nsteps) # 执行采样

    # 利用得到的后验概率模拟样本来估计链路的丢包率
    chain = sampler.get_chain(flat=True, discard=nsteps//2, thin=10)
    x_lossrate_estimated = np.zeors((n, 1))
    for i in range(ndim):
        x_lossrate_estimated[i] = np.percentile(chain[:, i], [75])

    # 根据估计的链路丢包率，获取链路的拥塞状态
    x_identified = (x_lossrate_estimated >= 0.01).astype(np.int8)

    return x_identified

if __name__ == '__main__':

    A_rm = np.array([[0,1,1,0,0,0,0,0,0,0,0], [0,1,0,1,1,0,0,0,0,0,0], [0,1,0,1,0,1,0,1,0,0,0], [0,1,0,1,0,1,1,0,1,0,0], [0,1,0,1,0,1,1,0,0,1,0], [0,1,0,1,0,1,1,0,0,0,1]], np.int8)

    m, n = A_rm.shape # 根据路由矩阵分别得到目标网络中路径与链路的数量

    num_times = 50

    x_pc = np.random.rand(n, 1)/5  # 链路拥塞概率在‘[0, 1/5]’之间, 列向量

    x_true = (np.random.rand(n, 1) <= x_pc).astype(np.int8)  # 获取链路的真实拥塞状态

    x_lossrate_true = np.random.rand(n, 1) * 0.01 + x_true * 0.05

    y_lossrate_true = np.zeros((m, 1)) # 路径状态观测值
    for j in range(m):
        y_lossrate_true[j] = get_path_likelihood(j, x_lossrate_true, A_rm)

    y = np.hstack((num_times * (1- y_lossrate_true), num_times * y_lossrate_true)).round()

    x_identified = alg_mcmc(y, A_rm) # 识别出的链路拥塞状态
