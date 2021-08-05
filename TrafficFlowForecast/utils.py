# -*- coding: utf-8 -*-
"""
对邻接矩阵进行归一化处理；
将邻接矩阵转变为稀疏张量；

"""
import tensorflow as tf
import scipy.sparse as sp
import numpy as np

from input_data import load_bj_stay_data,load_bj_data


def normalized_adj(adj):
    """
    :param adj: 交通区域的邻接矩阵
    :return:归一化后的邻接矩阵
    """
    adj = sp.coo_matrix(adj)  # 由于adj是稀疏矩阵类并且非常大，所以采用coo_matrix储存以减少内存占用
    rowsum = np.array(adj.sum(1))  # 邻接矩阵的行相乘获得和
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # rowsum开-1/2方后flatten折叠成一维的数组的用于数据标准化的d_inv_sqrt
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 将标准化参数d_inv_sqrt构成一个对角矩阵d_mat_inv_sqrt,同样是稀疏矩阵的储存方式
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # 数据类型统一转变为float32，节省内存同时尽量不影响精度
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj


def sparse_to_tuple(mx):
    """
    :param mx: 归一化后的稀疏矩阵
    :return:一个行主(row-major)排序的稀疏张量
    """
    mx = mx.tocoo()  # 转变为稀疏矩阵
    coords = np.vstack((mx.row, mx.col)).transpose()  # 获得坐标形式
    # print('mx',mx.data)
    L = tf.SparseTensor(coords, mx.data, mx.shape)  # 转变为
    # 将一个SparseTensor重新排序为规范的行主(row-major)排序.
    return tf.sparse_reorder(L)


def calculate_laplacian(adj, lambda_max=6):
    """
    :param adj: 交通区域的邻接矩阵
    :param lambda_max:调节流量留在本区域的权重
    :return:稀疏张量
    """
    # 不管流量是否能通向自己均存在流量停止不动的情况发生变化的，sp.eye(adj.shape[0])（单位矩阵）表示流量停止不动的情况。
    # 可以改变他的权重（当区域过大或者一个城市的交通情况很差时）
    adj = normalized_adj(adj + lambda_max * sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)  # 压缩稀疏矩阵的存储方式
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)


if __name__ == '__main__':
    # data, adj = load_bj_data('bj')
    data, adj = load_bj_stay_data('bj2')
    # adj1 = calculate_laplacian(adj)
    # print('normalized_adj',adj1)
    # normalized_adj(adj)
    print("calculate_laplacian",calculate_laplacian(adj))
