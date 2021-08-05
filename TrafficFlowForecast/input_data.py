# -*- coding: utf-8 -*-
"""
用于导入交通数据和划分训练集和测试集

adj:1024个区域之间的邻接矩阵（自己区域的流量不能进入自己区域）
adj2：1024个区域之间的邻接矩阵（自己区域的流量可以进入自己区域）
flow是预测的数据：流量
"""

import numpy as np
import pandas as pd
import pickle as pkl


def load_bj_data(dataset):
    """
    :param dataset: 数据集名称
    :return: 返回流量的历史记录和邻接矩阵
    """
    bj_adj = pd.read_csv(r'data/bj_adj_stay.csv', header=None)  # header=None忽略列名
    adj = np.mat(bj_adj)
    bj_tf = pd.read_csv(r'data/bj_flow.csv', header=None)
    return bj_tf, adj


def load_bj_stay_data(dataset):
    """
    :param dataset: 数据集名称
    :return: 返回流量的历史记录和邻接矩阵
    """
    bj_adj = pd.read_csv(r'data/bj_adj_stay.csv', header=None)  # header=None忽略列名
    adj = np.mat(bj_adj)
    bj_tf = pd.read_csv(r'data/bj_flow.csv', header=None)
    return bj_tf, adj


def load_merge_bj_data(dataset):
    """
    :param dataset: 数据集名称
    :return: 返回流量的历史记录和邻接矩阵
    """
    bj_adj = pd.read_csv(r'data/bj_merge_adj.csv', header=None)  # header=None忽略列名
    adj = np.mat(bj_adj)
    bj_tf = pd.read_csv(r'data/bj_merge_flow.csv', header=None)
    return bj_tf, adj


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    """
    预处理数据
    data：数据集
    time_len：数据的总的时间长度
    rate：划分出训练数据的比例
    seq_len：输入数据的时间长度
    pre_len：预测数据的时间长度
    return：trainX1, trainY1, testX1, testY1划分好的训练数据和测试数据
    """
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    # for i in range(int(len(test_data)/(seq_len + pre_len))):
    #     b = test_data[i*(seq_len + pre_len): i*(seq_len + pre_len) + seq_len + pre_len]
    #     testX.append(b[0: seq_len])
    #     testY.append(b[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1


if __name__ == '__main__':
    data, adj = load_bj_data('bj')
    # print("adj", adj)
    # print("data",data)
    time_len = data.shape[0]
    num_nodes = data.shape[1]
    data1 = np.mat(data, dtype=np.float32)
    print("时间长度是(单位30min)：",time_len)
    print("区域的个数是（观测区域）：",num_nodes)

    train_rate = 0.75
    seq_len = 11
    pre_len = 1
    max_value = np.max(data1)
    data1 = data1 / max_value
    trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)