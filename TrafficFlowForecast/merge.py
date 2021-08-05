#  coding: utf-8
"""
合并区域从而降低算法复杂度，将32*32个区域4个何为一个16*16

1024*1024的邻接矩阵转变为256*256的邻接矩阵，利用流量的区域相关性。

发现效果不是很好，于是没有使用！
"""
import copy
import xlsxwriter as xlsxwt

from write_adj import write_bj_adj_toRows, write_rows_toXlsx
from write_flow import readMat, readMergeRows, writeCsv, readMergeRows2

if __name__ == '__main__':
    """写入邻接矩阵"""
    # adj = write_bj_adj_toRows(8)
    # count = 0
    # for i in range(len(adj)):
    #     for j in range(len(adj[0])):
    #         if adj[i][j] == 1:
    #             count = count+1
    #             # print("坐标为：",i+1,j+1)
    # print(count)
    # write_rows_toXlsx("/Users/yuanbao/Desktop/test.xlsx", "data", adj)
    """写入流量矩阵"""
    trainD, testD = readMat()
    train_data = readMergeRows2(trainD)
    test_data = readMergeRows2(testD)
    data = train_data + test_data
    writeCsv("/Users/yuanbao/Desktop/bj_8x8_flow.csv", data)
