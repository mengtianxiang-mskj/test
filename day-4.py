import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
#每个批次的大小
batch_size = 100
#总共有多少批次
n_batch = mnist.train.num_examples // batch_size
#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

#初始化偏置
def bias_variable(shape):
    initial = tf.constant(0,1, shape= shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides = [1,1,1,1], padding ='SAME')

#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#定义两个占位符
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#改变x的格式
x_image = tf.reshape(x, [-1,28,28,1])
#初始化第一个卷积层的权值和偏置
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
#把x_image和权值的量进行卷积并加上偏置，再用relu
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+ b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#初始化第二个卷积层的权值和偏置
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
#把h_pool1和权值的量进行卷积并加上偏置，再用relu
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+ b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#把池化层的结果转化成一维
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
#初始化全连接层的权值和偏置
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+ b_fc1)
#定义keep_prob
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#初始化第二个全连接层的权值和偏置
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+ b_fc2 )
#交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
#优化器
train_step = tf.train.AdamOptimizer(0.0004).minimize(loss)
#初始化变量
init = tf.global_variables_initializer()
#结果放在一个布尔列表中
correct = tf.equal(tf.argmax(y, 1),tf.argmax(prediction, 1))
#准确率
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
#定义会话
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            print(batch_xs.shape)
            print(batch_ys.shape)
            sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
        acc = sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels,keep_prob:1})
        print('Iter'+ str(epoch)+',Testing accuracy'+ str(acc))