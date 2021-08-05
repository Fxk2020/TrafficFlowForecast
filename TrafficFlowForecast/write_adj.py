"""
写入32x32个区域之间的邻接矩阵

即1024个节点之间的邻接矩阵
"""
import copy
import numpy as np
import xlrd
import xlwt
import xlsxwriter as xlsxwt


def write_bj_adj_toRows(number):
    """
    :param number: 区域个数
    :return: number*number区域的邻接矩阵
    """
    row = list(range(number * number))
    rows = []
    index = number * number
    for i in range(index):
        for j in range(index):
            if j - i == number or i - j == number or j - i == 1 or i - j == 1:  # 所有边(有实际上不存在的边)
                row[j] = 1
            else:
                if j == i:
                    row[j] = 1  # 车辆可以留在原处，如果不能存在原处，则将该if语句去除掉
                else:
                    row[j] = 0
        rows.append(copy.deepcopy(row))
    for k in range(number - 1):
        """将实际上不存在的边去除掉"""
        less_index = (k + 1) * number
        rows[less_index - 1][less_index] = 0
        rows[less_index][less_index - 1] = 0

    return rows


def write_bj_adjMax_toRows(number):
    """
    :param number: 区域个数
    :return: number*number区域的邻接矩阵
    """
    row = list(range(number * number))
    rows = []
    index = number * number
    for i in range(index):
        for j in range(index):
            if j - i == number or i - j == number or j - i == 1 or i - j == 1 or \
                    j - i == (number - 1) or i - j == (number - 1) or \
                    j - i == (number + 1) or i - j == (number + 1):  # 所有边(有实际上不存在的边)
                row[j] = 1
            else:
                if j == i:
                    row[j] = 0  # 1  # 车辆可以留在原处，如果不能存在原处，则将该if语句去除掉
                else:
                    row[j] = 0
        rows.append(copy.deepcopy(row))
    for k in range(number - 1):
        """将实际上不存在的边去除掉32到33类似的边"""
        less_index = (k + 1) * number
        rows[less_index - 1][less_index] = 0
        rows[less_index][less_index - 1] = 0
    for k in range(number):
        """将实际上不存在的边去除掉1到32类似的边"""
        less_index2 = k * number + 1
        rows[less_index2-1][less_index2+31-1] = 0
        rows[less_index2+31-1][less_index2-1] = 0
    for k in range(number-2):
        """将实际上不存在的边去除掉32到65类似的边"""
        less_index3 = (k+1) * number
        rows[less_index3-1][less_index3+33-1] = 0
        rows[less_index3+33-1][less_index3-1] = 0

    return rows


def write_rows_toXlsx(path, sheet_name, value):
    """将临界矩阵写入xlsx中"""
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlsxwt.Workbook(path)  # 新建一个工作簿
    worksheet = workbook.add_worksheet(sheet_name)  # 工作区间
    for i in range(index):
        for j in range(len(value[i])):
            worksheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.close()
    print("xlsx格式表格写入数据成功！")


if __name__ == '__main__':
    adj = write_bj_adjMax_toRows(32)
    count = 0
    for i in range(len(adj)):
        for j in range(len(adj[0])):
            if adj[i][j] == 1:
                count = count + 1
                # print("坐标为：",i+1,j+1)
    print(count)
    write_rows_toXlsx("/Users/yuanbao/Desktop/test.xlsx", "data", adj)
