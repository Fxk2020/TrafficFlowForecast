# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

from input_data import load_bj_stay_data
from utils import calculate_laplacian


class tgcnCell(RNNCell):
    """
    继承自RNN 单元的抽象对象
    实现Temporal Graph Convolutional Network--TGCN
    """
    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):
        """
        num_units:gru_units门递归网络的数目
        adj：表示区域之间空间关系的邻接矩阵
        num_nodes：一共有几个区域
        act：激活函数使用tanh
        """
        super(tgcnCell, self).__init__(_reuse=reuse)
        print("t-gcn正在初始化......................")
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._adj = []
        # 对邻接矩阵的数据进行正规化
        self._adj.append(calculate_laplacian(adj))
        print("t-gcn初始化完毕......")

    @property
    def state_size(self):
        """单元使用的状态的大小"""
        return self._nodes * self._units

    @property
    def output_size(self):
        """单元产生的输出大小"""
        return self._units

    def __call__(self, inputs, state, scope=None):
        """
        gru部分的实现
        """
        # 变量共享variable_scope
        with tf.variable_scope(scope or "tgcn"):
            with tf.variable_scope("gates"):
                # sigmoid作为激活函数，进行图卷积
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope)
                )
                # 重置门r用于控制先前时刻状态信息的度量，上传门u用于控制上传到下一状态的信息度量
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)  # 分割卷积后的张量
            with tf.variable_scope("candidate"):
                # candidate部分对应公式Ct
                r_state = r * state
                # tanh作为激活函数，进行图卷积
                c = self._act(
                    self._gc(inputs, r_state, self._units, scope=scope)
                )
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        """
        gcn部分的实现
        """
        """inputs:(-1,num_nodes)"""
        inputs = tf.expand_dims(inputs, 2)
        """state:(batch,num_node,gru_units)"""
        state = tf.reshape(state, (-1, self._nodes, self._units))
        """concat"""
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.get_shape()[2].value
        """(num_node,input_size,-1)"""
        x0 = tf.transpose(x_s, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._nodes, -1])

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            for m in self._adj:
                # print(m)
                x1 = tf.sparse_tensor_dense_matmul(m, x0)
                # print(x1)
            x = tf.reshape(x1, shape=[self._nodes, input_size, -1])
            x = tf.transpose(x, perm=[2, 0, 1])
            x = tf.reshape(x, shape=[-1, input_size])
            weights = tf.get_variable(
                'weights', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)
            biases = tf.get_variable(
                "biases", [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
            # 激活得到两层GCN，对应卷积过程公式F
            x = tf.nn.bias_add(x, biases)
            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x


# if __name__ == '__main__':
#     gru_units = 64
#     data, adj = load_bj_stay_data('bj2')
#     time_len = data.shape[0]
#     num_nodes = data.shape[1]
#     data1 = np.mat(data, dtype=np.float32)
#     cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
