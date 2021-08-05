import tensorflow as tf
import os

"""
tensorflow1.15.0的学习笔记
"""

# 屏蔽部分TensorFlow的warning信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 初始化图和输出路径
tf.compat.v1.reset_default_graph()
logdir = "logs/"

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
# print(node1, node2)

# 创建了一个Session对象，并且执行了它的run方法来运行包含了node1和node2的computational graph的计算结果
sess = tf.Session()
# print(sess.run([node1, node2]))

# 占位符--相当于形式参数
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + 是tf.add(a, b)的缩写形式

# print(sess.run(adder_node, {a: 3, b:4.5}))
# print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

add_and_triple = adder_node * 3.
# print(sess.run(add_and_triple, {a: 3, b:4.5}))

# 画出计算图

# 可以训练的参数Variable
W = tf.Variable([.3], tf.float32)  # 参数并没有初始化
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# 初始化所有的全局变量
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# 损失函数
"""损失函数会计算出当前训练出的模型和所提供的数据之间的距离。
我们将使用用于线性回归的标准损失模型，其原理是将当前模型和提供的数据之间的增量的平方求和。
linear_model - y创建一个向量，其中每个元素是相应的样本的误差增量。我们称之为tf.square平方误差。
然后，我们使用tf.reduce_sum将所有平方误差求和，以创建一个单一的标量，用于提取出表示所有样本的总误差值："""
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print("参数优化前的误差：",sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 更改W和b的值--猜测出最优参数
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print("最优参数的误差：",sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 通过优化器进行训练
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)  # 将值重置为不正确的默认值
for i in range(10000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# print(sess.run([W, b]))
fixW = tf.assign(W, W)
fixb = tf.assign(b, b)
sess.run([fixW,fixb])
print("训练后优化参数的误差：",sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())
writer.close()