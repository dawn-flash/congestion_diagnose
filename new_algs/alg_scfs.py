'''
[IEEE ToIT'06] Network Tomography of Binary Network Performance Characteristics
'''

import numpy as np
import copy


def alg_scfs(y: np.ndarray, A_rm: np.ndarray):
    """
    scfs算法的调用接口
    :param y:路径的观测状态，列向量；如为矩阵，横维度为时间，纵维度为链路状态
    :param A_rm:为路由矩阵，纵纬度为路径，横维度为链路
    :return: x_identified 列向量，如为矩阵，横维度为时间，纵维度为链路推测状态
    """

    if np.ndim(y) <= 1:  # 强制转换为列向量
        y = y.reshape((-1, 1))

    _, n = A_rm.shape
    _, num_time = y.shape

    x_identified = np.zeros((n, num_time),dtype=int)
    y_t = y.transpose()
    for i in range(num_time):
        path_state_obv = y_t[i]
        links_state_infered = SCFS(path_state_obv, A_rm)
        for j in range(links_state_infered.shape[0]):
            x_identified[j][i] = links_state_infered[j]

    return x_identified


def SCFS(y: np.ndarray, A_rm: np.ndarray):
    """
    根据一次路径观测状态，进行SCFS算法推断
    :param y: 观测的路径状态
    :param A_rm: 路由矩阵
    :return:
    """
    tree_vector = rm_to_vector(A_rm)
    num_links = len(tree_vector)

    links_state_inferred = np.zeros(num_links + 1)  # 链路初始诊断为正常状态
    paths_state_obv_temp = copy.deepcopy(y)  # 设置一个临时变量
    algorithm(1, A_rm, paths_state_obv_temp, links_state_inferred)  # 从链路 l_1 开始诊断

    links_state_inferred = links_state_inferred[1:]  # 去掉根节点0所在的虚拟链路状态

    return links_state_inferred


def algorithm(k, route_matrix: np.ndarray, paths_state_obv_temp, links_state_inferred: np.ndarray):
    """
    递归算法
    :param k: 对以k结点为根的子树进行递归
    :param route_matrix: 路由矩阵
    :param paths_state_obv_temp: 观测的路径状态
    :param links_state_inferred: 推测的链路状态
    :return:
    """
    tree_vector = rm_to_vector(route_matrix)
    leaf_nodes = get_leaf_nodes(tree_vector)
    if k not in leaf_nodes:
        d = get_children(k, tree_vector)

        path, = np.where(route_matrix[:, k - 1] > 0)
        links_state_inferred[k] = np.min(paths_state_obv_temp[path])

        if links_state_inferred[k]:
            links_state_inferred[k] = 1  # 强制设为布尔状态
            paths_state_obv_temp[path] = 0  # 将路径状态重置为0

        for i in d:
            algorithm(i, route_matrix, paths_state_obv_temp, links_state_inferred)  # 递归操作

    else:
        path = get_paths(k, route_matrix)
        links_state_inferred[k] = paths_state_obv_temp[path]

        if links_state_inferred[k]:
            links_state_inferred[k] = 1  # 强制设为布尔状态

        paths_state_obv_temp[path] = 0  # 将所有路径状态重置为0


def rm_to_vector(A_rm: np.ndarray):
    """
    将路由矩阵转换为树向量
    :param A_rm:
    :return:
    """
    # a=A_rm.shape[1]
    tree_vector = [0] * (A_rm.shape[1])

    for i in range(A_rm.shape[0]):
        path = A_rm[i]
        pre_node = 0
        for j in range(path.shape[0]):
            if path[j] == 1:
                tree_vector[j] = pre_node
                pre_node = j + 1

    return tree_vector


def get_children(node: int, tree_vector: list):
    """
    获取结点k的所有孩子结点
    :param node: 指定的结点
    :return: children：list
    """
    children = []
    leaf_nodes = get_leaf_nodes(tree_vector)
    if node not in leaf_nodes:
        for index, item in enumerate(tree_vector):
            if item == node:
                children.append(index + 1)
    return children


def get_leaf_nodes(tree_vector: list):
    """
    树的所有叶子结点
    :param tree_vector list
    :return: leaf_nodes:np.ndarray
    """
    leaf_nodes = []
    for index in range(len(tree_vector)):
        leaf_node = index + 1
        if leaf_node not in tree_vector:
            leaf_nodes.append(leaf_node)
    return np.array(leaf_nodes)


def get_paths(link: int, route_matrix: np.ndarray):
    """
    获取进过链路link的所有路径
    :param link:
    :return: paths:list
    """
    assert link > 0
    paths, = np.where(route_matrix[:, link - 1] > 0)
    return paths.tolist()


def test():
    p = [1, 2, 3]
    a = np.array([1, 2, 3, 4, 5])
    a[p] = 0
    print(a)


def test_rm():
    # A_rm=np.array([[1,1,0,0,0],
    #                [1,0,1,1,0],
    #                [1,0,1,0,1]])

    A_rm = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                     [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, ],
                     [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                     [1, 0, 1, 0, 1, 1, 0, 0, 0, 1]])
    tree_vector = rm_to_vector(A_rm)
    print(tree_vector)


def test_scfs():
    # y = np.zeros((3, 1))
    y=np.array([0,1,0]).reshape((-1,1))
    # y=np.array([[0,1,0],[1,1,1],[1,0,1]]).transpose()
    A_rm = np.array([[1, 1, 0, 0, 0],
                     [1, 0, 1, 1, 0],
                     [1, 0, 1, 0, 1]])
    links_state_infered = alg_scfs(y, A_rm)
    print(links_state_infered)


if __name__ == '__main__':
    # test()
    # test_rm()
    test_scfs()
