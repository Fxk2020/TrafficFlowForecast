#  coding: utf-8
"""
对运行结果进行可视化
"""

import matplotlib.pyplot as plt


def plot_result(test_result, test_label1, path,show):
    """
    预测结果可视化
    :param test_result:1024个区域预测的流量值
    :param test_label1:1024个区域真实的流量值
    :param path:保存分析结果的路径
    :param show: 运行完程序后是否显示图像
    """
    """区域i--all test result visualization"""
    i = 2
    fig1 = plt.figure(figsize=(8, 2.5))
    #    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[:, i]
    a_true = test_label1[:, i]
    plt.plot(a_pred, 'r-', label='prediction')
    plt.plot(a_true, 'b-', label='true')
    plt.legend(loc='best', fontsize=10)
    plt.savefig(path + '/test_all.jpg')
    if show:
        plt.show()
    """区域i--oneday test result visualization"""
    fig1 = plt.figure(figsize=(8, 2.5))
    #    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[0:48, i]
    a_true = test_label1[0:48, i]
    plt.plot(a_pred, 'r-', label="prediction")
    plt.plot(a_true, 'b-', label="true")
    plt.legend(loc='best', fontsize=10)
    plt.savefig(path + '/test_oneday.jpg')
    if show:
        plt.show()
    """区域i--12h test result visualization"""
    # fig1 = plt.figure(figsize=(7, 1.5))
    # #    ax1 = fig1.add_subplot(1,1,1)
    # a_pred = test_result[0:24, i]
    # a_true = test_label1[0:24, i]
    # plt.plot(a_pred, 'r-', label="prediction")
    # plt.plot(a_true, 'b-', label="true")
    # plt.legend(loc='best', fontsize=10)
    # plt.savefig(path + '/test_12h.jpg')
    # if show:
    #     plt.show()


def plot_error(train_rmse, train_mae, test_rmse, test_acc, test_mae, path,show):
    """
    显示训练过程中监控指标的变化
    :param train_rmse:训练中的rmse--需要的
    :param train_mae:训练中的mae--需要的
    :param test_rmse:测试中的rmse
    :param test_acc:
    :param test_mae:测试中的mae
    :param path:保存分析结果的路径
    :param show: 运行完程序后是否显示图像
    :return:
    """
    """train_rmse & test_rmse"""
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best', fontsize=10)
    plt.savefig(path + '/rmse.jpg')
    if show:
        plt.show()
    """train_mae & test_mae"""
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(train_mae, 'r-', label="train_mae")
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best', fontsize=10)
    plt.savefig(path + '/mae.jpg')
    if show:
        plt.show()

    """accuracy"""
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best', fontsize=10)
    plt.savefig(path + '/test_acc.jpg')
    if show:
        plt.show()
