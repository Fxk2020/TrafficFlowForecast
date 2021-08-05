# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import numpy.linalg as la
from input_data import preprocess_data, load_bj_data, load_bj_stay_data, load_merge_bj_data
from tgcn import tgcnCell
# from gru import GRUCell
from visualization import plot_result, plot_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import matplotlib.pyplot as plt
import time
import os
import pytz
import datetime

"""屏蔽部分TensorFlow的warning信息"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

time_start = time.time()

"""记录结果的文件"""
# 由时间戳获取北京时间
bj_time = datetime.datetime.fromtimestamp(int(time_start), pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
bj_time = bj_time.replace(":", "-")  # Windows下路径名中不能含":"
# 创建文件独一无二的路径
resultFile = open(r'/Users/yuanbao/PycharmProjects/TrafficFlowForecast/out/result/result'+str(bj_time)+'.txt','w')

"""
        Settings:超参和全局变量设计
"""
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 32, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 20, '  time length of inputs.')
flags.DEFINE_integer('pre_len', 1, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.75, 'rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_integer('earlyStopping', 15,'Number of epochs to stop if acc is not add')
flags.DEFINE_string('dataset', 'bj2', 'bj ,bj2 or mergeBj.')  # bj没有设置自己邻接；bj2设置了自己邻接
flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
flags.DEFINE_boolean('show',False,'True or False')
flags.DEFINE_string('logdir','logs','logs/i/')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
earlyStopping = FLAGS.earlyStopping
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units
show = FLAGS.show
logdir = FLAGS.logdir

"""
load data : 加载数据
"""
if data_name == 'bj':
    data, adj = load_bj_data('bj')
if data_name == 'bj2':
    data, adj = load_bj_stay_data('bj2')
if data_name == 'mergeBj':
    data, adj = load_merge_bj_data('mergeBj')

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 = np.mat(data, dtype=np.float32)

"""
normalization：使数据位于0~1之间， 防止溢出；
第一种方式 最简单的处理数据/maxValue
第二种方式 (a-min)/(max-min)
"""
max_value = np.max(data1)
min_value = np.min(data1)
data1 = data1 / max_value
# data1 = (data1-min_value) / (max_value-min_value)
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0] / batch_size)  # 一个需要多少个批次模型才能训练完成
training_data_count = len(trainX)


def TGCN(_X, _weights, _biases):
    """
    构建TGCN模型
    :param _X:训练数据
    :param _weights:通往gru的权重
    :param _biases:偏执
    :return:预测值output，最终的状态states
    """
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    # 把多个rnn类的单元-cell_1连在一起，构建一个多层的网络结构
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    # static_rnn创建一个由RNNCell指定的递归神经网络cell；states返回的最终的状态
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
        o = tf.reshape(o, shape=[-1, gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    return output, m, states


"""初始化计算图--用于tensorboard绘制"""
tf.compat.v1.reset_default_graph()

"""Graph placeholders: 使用占位符来接受外部的输入"""
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

"""Graph Variables：允许我们向graph中添加可训练的参数。网络权重和偏执的初始化用于训练"""
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([pre_len]), name='bias_o')}

"""初始化模型"""
if model_name == 'tgcn':
    pred, ttts, ttto = TGCN(inputs, weights, biases)
y_pred = pred

"""optimizer 训练中的优化器和rmse、mae"""
lambda_loss = 0.0015
# 防止过拟合权值衰减--L2正则化项
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
# loss 避免出现过拟合
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + Lreg)
# rmse reduce_mean用于求平均值
error_rmse = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
# MAE
error_mae = tf.reduce_mean(tf.abs(y_pred - label))
# 优化器
# AdamOptimizer是TensorFlow中实现Adam算法的优化器。
# Adam即Adaptive Moment Estimation（自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正。
# Adam 算法相对于其它种类算法有一定的优越性，是比较常用的算法之一。
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

"""使用模型训练参数信息创建目录"""
variables = tf.global_variables()
# 保存模型的具体参数
saver = tf.train.Saver(tf.global_variables())
out = 'out/%s' % (model_name)
# out = 'out/%s_%s'%(model_name,'perturbation')
# 以模型的参数命名文件夹
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r' % (
    model_name, data_name, lr, batch_size, gru_units, seq_len, pre_len, training_epoch)
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)

"""创建计算图会话"""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  # 限制使用GPU
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())  # 初始化所有的参数


def evaluation(a, b):
    """
    评估模型
    :param a: 真实值
    :param b: 预测值
    :return:rmse, mae, 1 - F_norm
    """
    # print(b.shape)
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
    accuracy = 1 - F_norm
    return rmse, mae, accuracy


x_axe, batch_loss, batch_rmse, batch_mae, batch_pred = [], [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_pred = [], [], [], [], []

"""训练模型"""
print("开始训练模型....................",file=resultFile)


def IsOverFitting(acc, earlyStopping,minEpoch):
    """
    判断是否发生过拟合
    :param acc: 准确率
    :param earlyStopping:容忍acc不增加的轮数
    :param minEpoch:最少多少轮后进行早停判断
    :return: 是否需要早停
    """
    if len(acc) < minEpoch:
        return False
    else:
        for i in range(earlyStopping - 1):
            index = -(i + 1)
            if acc[index] > acc[index - 1]:
                return False

    return True


for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size: (m + 1) * batch_size]
        mini_label = trainY[m * batch_size: (m + 1) * batch_size]
        _, loss1, rmse1, mae1, train_output = sess.run([optimizer, loss, error_rmse, error_mae, y_pred],
                                                       feed_dict={inputs: mini_batch, labels: mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)
        batch_mae.append(mae1 * max_value)
        # batch_rmse.append(rmse1 * (max_value-min_value))
        # batch_mae.append(mae1 * (max_value-min_value))

    # 在每个epoch都Test completely
    loss2, rmse2, mae2, test_output = sess.run([loss, error_rmse, error_mae, y_pred],
                                               feed_dict={inputs: testX, labels: testY})
    # print("testX",testX.shape)  # (324, 9, 1024)
    # print("test_output",test_output.shape)  # (972, 1024)
    test_label = np.reshape(testY, [-1, num_nodes])
    rmse, mae, acc = evaluation(test_label, test_output)  # 计算预测值和真实值之间的差距
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    # test_label1 = test_label * (max_value-min_value)
    # test_output1 = test_output * (max_value-min_value)
    test_loss.append(loss2)
    test_rmse.append(rmse*max_value)
    test_mae.append(mae*max_value)
    # test_rmse.append(rmse * (max_value-min_value))
    # test_mae.append(mae * (max_value-min_value))
    test_acc.append(acc)
    test_pred.append(test_output1)

    # 用于控制板输出
    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'train_mae:{:.4}'.format(batch_mae[-1]),
          'test_rmse:{:.4}'.format(test_rmse[-1]),
          'test_mae:{:.4}'.format(test_mae[-1]),
          'test_acc:{:.4}'.format(test_acc[-1]))

    # 自己实现早停防止过拟合
    if IsOverFitting(test_acc, earlyStopping,minEpoch=10):
        break

"""写入计算图--用于tensorboard绘制"""
writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())
writer.close()

"""visualization 运行结果可视化"""
b = int(len(batch_rmse) / totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i * totalbatch:(i + 1) * totalbatch]) / totalbatch) for i in range(b)]
batch_mae1 = [i for i in batch_mae]
train_mae = [(sum(batch_mae1[i * totalbatch:(i + 1) * totalbatch]) / totalbatch) for i in range(b)]

"""寻找训练过程中最小的rmse进行可视化"""
index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
# print('test_result',test_result.shape)  # (972, 1024)
# print('test_label1',test_label1.shape)  # (972, 1024)
plot_result(test_result,test_label1,path,show)
plot_error(train_rmse,train_mae,test_rmse,test_acc,test_mae,path,show)

"""输出结果参数--写入文件"""
print('min_rmse:%r' % (np.min(batch_rmse)),file=resultFile)
print('min_mae:%r' % (np.min(batch_mae)),file=resultFile)
print('max_acc:%r' % (np.max(acc)),file=resultFile)

time_end = time.time()
print("程序的运行时间为：", time_end - time_start, 's',file=resultFile)
resultFile.close()
