import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess=tf.Session()
a=tf.constant(1)
b=tf.constant(3)
print(sess.run(a+b))
from tensorflow.examples.tutorials.mnist import input_data 
  
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  # get the data source
mnist_data_folder="C:\\Users\\Administrator\\mnist\\datasets"
mnist = input_data.read_data_sets(mnist_data_folder,one_hot=True)
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# input image:pixel 28*28 = 784
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder('float', [None, 10])  # y_ is realistic result
    
with tf.name_scope('image'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # any dim, width, height, channel(depth)
    tf.summary.image('input_image', x_image, 8)
    # the first convolution layer
 
with tf.name_scope('conv_layer1'):
    W_conv1 = weight_variable([5, 5, 1, 32])  # convolution kernel: 5*5*1, number of kernel: 32
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # make convolution, output: 28*28*32
    
with tf.name_scope('pooling_layer'):
    h_pool1 = max_pool_2x2(h_conv1)  # make pooling, output: 14*14*32

# the second convolution layer
with tf.name_scope('conv_layer2'):
    W_conv2 = weight_variable([5, 5, 32, 64])  # convolution kernel: 5*5, depth: 32, number of kernel: 64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output: 14*14*64
 
with tf.name_scope('pooling_layer'):
    h_pool2 = max_pool_2x2(h_conv2)  # output: 7*7*64


 # the first fully connected layer
with tf.name_scope('fc_layer3'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])  # size: 1*1024
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # output: 1*1024
    
 # dropout
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

 
 # the second fully connected layer
 # train the model: y = softmax(x * w + b)
with tf.name_scope('output_fc_layer4'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])  # size: 1*10

with tf.name_scope('softmax'):
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # output: 1*10

with tf.name_scope('lost'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    tf.summary.scalar('lost', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_summary = tf.summary.FileWriter(r'C:\Users\Administrator\tf\log', tf.get_default_graph())
 
# init all variables
init = tf.global_variables_initializer()

# run session
with tf.Session() as sess:
    sess.run(init)
    # train data: get w and b
    for i in range(2000):  # train 2000 times
        batch = mnist.train.next_batch(50)

        result, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
         # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        if i % 100 == 0:
             # train_accuracy = sess.run(accuracy, feed_dict)
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})  # no dropout
            print('step %d, training accuracy %g' % (i, train_accuracy))
 
             # result = sess.run(merged, feed_dict={x: batch[0], y_: batch[1]})
            train_summary.add_summary(result, i)
 
    train_summary.close()
 
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
 
 
 #  open tensor_board in windows-cmd
 #  tensorboard --logdir=C:\Users\Administrator\tf
