#!/usr/bin/env python

from cifar10_utils import get_cifar10
import tensorflow as tf
import numpy as np

BATCH_SIZE = 1000
IMAGE_SIZE = 32
NUMBER_OF_CLASSES = 10

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def translate_labels(lab):
	trlab = np.zeros((len(lab),NUMBER_OF_CLASSES))
	for i,l in enumerate(lab):
		trlab[i][l] = 1	
	return trlab

def create_batch(num_batch,batchsize,data_list,labels_list):
	# 0 is the first index
	beg_ind = (num_batch * batchsize) % labels_list.shape[0]
	end_ind = ((num_batch + 1) * batchsize) %labels_list.shape[0]
	if (end_ind < batchsize):
		beg_ind = 0
		end_ind = batchsize
	batch_list = []
	batch_list.append(data_list[beg_ind:end_ind,0:(IMAGE_SIZE*IMAGE_SIZE)])
	batch_list.append(translate_labels(labels_list[beg_ind:end_ind]))
	return batch_list


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE*IMAGE_SIZE*1])
y_ = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])

keep_prob = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
W_fc1 = weight_variable([8*8*64,1024])
b_fc1 = bias_variable([1024])
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

cifar10_dict = get_cifar10()

sess.run(tf.global_variables_initializer())

x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(20000):
	batch = create_batch(i,BATCH_SIZE,cifar10_dict['data'],cifar10_dict['labels'])
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1], keep_prob:1.0})
		print("step %d, training accuracy %g"%(i,train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob:0.5})


print("test accuracy %g"%accuracy.eval(feed_dict={x: cifar10_dict['data_test'][:,0:IMAGE_SIZE], y_:cifar10_dict['labels_test'],keep_prob:1.0}))
