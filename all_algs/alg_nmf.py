import numpy as np
from sklearn.decomposition import NMF


def alg_cvxfd(y: np.ndarray, A_rm: np.ndarray,threshold:float=0.01):
    # y 为观测到的路径拥塞状态，列向量；如为矩阵，横维度为时间；
    # A_rm 为“routing matrix”，矩阵：纵维度为路径，横维度为链路；
    # x_pc 为“probability of congestion”，列向量；
    # sigma 为测量噪声的平方差，标量；
    # x_identified 为返回的已识别的链路拥塞状态，列向量；如为矩阵，横维度为时间

    m, n = A_rm.shape  # 分别获取目标网络的路径和链路的总数量
    _, num_times = y.shape  # 获得不同的时刻数目

    x_identified = np.zeros((n, num_times))  # 识别出的链路拥塞状态，列向量；如为矩阵，横维度为时间


    x_identified = init_nmf(n,y)

    new_x_identified=cal_links_state_infer(x_identified,threshold)

    return new_x_identified

def cal_links_state_infer(links_loss_infer:np.ndarray,threshold):
    """
    根据链路的丢包率和丢包率门限，计算链路的拥塞状态
    :param links_loss_infer: array(n,s) s次实验得到的链路丢包率
    :return:  links_state_infer(n,s)
    """
    links_state_infer=np.zeros(shape=links_loss_infer.shape)
    for i in range(links_loss_infer.shape[0]):
        for j in range(links_loss_infer.shape[1]):
            if links_loss_infer[i][j]>=threshold:
                links_state_infer[i][j]=1
            else:
                links_state_infer[i][j]=0

    return links_state_infer



def init_nmf(num_links: int, y: np.ndarray):
    """
    nmf算法的核心部分
    :param num_links: int  链路的数量
    :param y: array(m,s)  s次实验得到的路径的丢包率列表
    :return:  array(n,s)  s次实验得到的链路丢包率列表
    """
    nmf = NMF(n_components=num_links,  # 链路总数量
                     init="random",  # 初始化，"nndsvd" 更适用于稀疏矩阵
                     solver="cd",  # "cd" 为 Coordinate Descent Solver 坐标下降处理器，"cd" 只能优化 Frobenius norm 函数
                     beta_loss="frobenius",  # frobenius norm
                     tol=1e-4,  # 停止迭代的条件
                     max_iter=20000,  # 最大迭代次数
                     random_state=None,  # 随机状态
                     alpha=0.,  # 正则化参数
                     l1_ratio=0.,  # 正则化混合参数
                     verbose=0,
                     shuffle=False,  # 针对 "cd" 处理器
                     regularization="both",  #
                     )
    nmf.fit(y)
    return nmf.components_



# from all_DS.config import Config
# import os
# from all_tools import utils
# def test_large_scale():
#     # file_path=os.path.join(os.path.dirname(os.getcwd()),'all_DS','TOPO_DS','[0, 1, 1, 3, 3].json')
#     file_path = os.path.join(os.path.dirname(os.getcwd()), 'all_DS', 'TOPO_DS', '[0, 1, 1, 3, 3, 5, 5, 6, 6, 6].json')
#
#     config = Config(file_path)
#     path_loss_rate_matrix = config.get_paths_loss_rate('[0, 0.1]')
#     links_state_true = config.get_links_state_true('[0, 0.1]')
#     routing_matrix = config.get_routing_matrix()
#
#     res = alg_cvxfd(y=path_loss_rate_matrix, A_rm=routing_matrix)
#     print(res)
#     dr, fpr, f1, j = utils.get_drfpr_f1j(links_state_true, res)
#     print("dr", dr)
#     print("fpr", fpr)
#     print("f1", f1)
#     print("j", j)

if __name__ == '__main__':

    # test_large_scale()
    pass
