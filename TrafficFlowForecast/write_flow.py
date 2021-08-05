# coding:UTF-8
"""
读入矩阵文件，并将每个区域流量随时间的变化写入文件
"""

import scipy.io as scio
import numpy as np
import csv
import copy
import pandas as pd


def readMat(url='/Users/yuanbao/Desktop/机器学习平台及其应用实践/实验！！60分/基于城市网络的交通流量预测/题目/data.mat'):
    """
    返回读取的mat文件为两个矩阵train和test
    :param url: 文件路径
    :return: train(32, 32, 1008)和test矩阵 (32, 32, 336)
    """
    dataFile = url
    data = scio.loadmat(dataFile)

    trainD = data['train']
    testD = data['test']

    return trainD, testD


def readRows(train):
    """
    对数据集按每个区域的所有时间的流量进行格式化
    :param train: 流量数据
    :return: 整形后的流量
    """
    shape = train.shape
    row = []
    rows = []
    for i in range(shape[2]):
        for j in range(shape[1]):
            for k in range(shape[0]):
                # print(train[k, j, i],end=" ")  # k是行数，j是列数，i是矩阵数
                row.append(train[k, j, i])
        rows.append(copy.deepcopy(row))
        row = []
    return rows


def readMergeRows(train):
    """
    对数据集按合并后区域的所有时间的流量进行格式化
    :param train: 流量数据
    :return: 整形后的流量
    """
    shape = train.shape
    row = []  # 池化
    row2 = []  # 未池化
    rows = []
    for i in range(shape[2]):
        for j in range(shape[1]):
            for k in range(shape[0]):
                # print(train[k, j, i],end=" ")  # k是行数，j是列数，i是矩阵数
                row.append(train[k, j, i])
        # row含有1024个区域的i时刻的流量数据，对其进行压缩
        for o in range(16):
            for p in range(16):
                p = 2*p+64*o
                row2.append(row[p]+row[p+1]+row[p+32]+row[p+33])
        # print(row)
        # print(len(row2))
        rows.append(copy.deepcopy(row2))
        row = []
        row2 = []
    return rows


def readMergeRows2(train):
    """
    对数据集按合并后区域的所有时间的流量进行格式化
    :param train: 流量数据
    :return: 整形后的流量
    """
    shape = train.shape
    row = []  # 池化
    row2 = []  # 未池化
    rows = []
    for i in range(shape[2]):
        for j in range(shape[1]):
            for k in range(shape[0]):
                # print(train[k, j, i],end=" ")  # k是行数，j是列数，i是矩阵数
                row.append(train[k, j, i])
        # row含有1024个区域的i时刻的流量数据，对其进行压缩
        for o in range(8):
            for p in range(8):
                p = 4*p+128*o
                row2.append(row[p]+row[p+1]+row[p+2]+row[p+3]
                            +row[p+32]+row[p+33]+row[p+34]+row[p+35]
                            +row[p+64]+row[p+64+1]+row[p+64+2]+row[p+64+3]
                            +row[p+96]+row[p+96+1]+row[p+96+2]+row[p+96+3])
        # print(row)
        # print(len(row2))
        rows.append(copy.deepcopy(row2))
        row = []
        row2 = []
    return rows


def writeCsv(path,rows):
    """
    将格式化的数据写入到csv文件中
    :param path:
    :param rows:
    :return:
    """
    with open(path, "w") as csvfile:
        writer = csv.writer(csvfile)
        # 写入多行用writerows,写入单行使用writerow
        for i in range(len(rows)):
            writer.writerow(rows[i])


def showDate():
    train, test = readMat()
    train_data = readRows(train)
    test_data = readRows(test)
    data = train_data+test_data
    writeCsv("/Users/yuanbao/Desktop/bj_merge_flow.csv",data)


def load_csv_data():
    bj_adj = pd.read_csv(r'/Users/yuanbao/Desktop/工作簿3.csv', header=None)  # header=None忽略列名
    return bj_adj


if __name__ == '__main__':
    trainD, testD = readMat()
    print(trainD.shape)
    print(testD.shape)
    # road = load_csv_data()
    # print(road.values[0][j*32+k])
    # print(road.shape)
    # train_data = readMergeRows2(trainD)
    # writeCsv("/Users/yuanbao/Desktop/bj_merge_flow.csv", train_data)
    # # trainD.shape[2]

    for i in range(trainD.shape[2]):
        for j in range(trainD.shape[1]):
            for k in range(trainD.shape[0]):
                print("第" + str(i + 1) + "天的数据",'区域'+str(j*32+k+1)+'流量为:',trainD[k,j,i])

    # for i in range(testD.shape[2]):
    #     for j in range(testD.shape[1]):
    #         for k in range(testD.shape[0]):
    #             print("第" + str(i + 1 +1008) + "天的数据",'区域'+str(j*32+k+1)+'流量为:',testD[k,j,i])
