import tensorflow as tf

import cifar10_input

tf.logging.set_verbosity(tf.logging.INFO)


image_size  = cifar10_input.IMAGE_SIZE
num_classes = cifar10_input.NUM_CLASSES
num_epochs  = 200
batch_size  = 128
num_examples_per_epoch_for_train = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_examples_per_epoch_for_eval  = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

num_epochs_per_decay       = 50.0
learning_rate_decay_factor = 0.1
initial_learning_rate      = 0.1

#data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# the following functions are based on code from:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# modified by Nils Kornfeld
#
#
#BEGIN APACHE LICENSED CODE
def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing

    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zerof_fraction(x))

def distorted_inputs():
    data_dir = os.path.join('cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    return images, labels

def inputs(eval_data):
    data_dir = os.path.join('cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=1)
    return images, labels
#END APACHE LICENSED CODE


def conv_norm_block(data, n_filters, filter_size):
    tmptens = tf.layers.conv2d(data, n_filters, filter_size, padding="same")
    tmptens = tf.layers.batch_normalization(tmptens, training=is_training)
    return tf.nn.relu(tmptens)

def res_block_2015(data, n_filters, filter_size):
    tmptens = conv_norm_block(data, n_filters, filter_size)
    tmptens = tf.layers.conv2d(tmptens, n_filters, filter_size, padding="same")
    tmptens = tf.layers.batch_normalization(tmptens, training=is_training)
    tmptens = tf.add(data, tmptens)
    return tf.nn.relu(tmptens)

def res_block_2016(data, n_filters, filter_size):
    tmptens = tf.layers.batch_normalization(data, training=is_training)
    tmptens = tf.nn.relu(tmptens)
    tmptens = tf.layers.conv2d(tmptens, n_filters, filter_size, padding="same")
    tmptens = tf.layers.batch_normalization(tmptens, training=is_training)
    tmptens = tf.nn.relu(tmptens)
    tmptens = tf.layers.conv2d(tmptens, n_filters, filter_size, padding="same")
    tmptens = tf.add(data, tmptens)
    return tmptens

def no_relu(data, n_filters, filter_size):
    tmptens = conv_norm_block(data, n_filters, filter_size)
    tmptens = tf.layers.conv2d(tmptens, n_filters, filter_size, padding="same")
    tmptens = tf.layers.batch_normalization(tmptens, training=is_training)
    tmptens = tf.add(data, tmptens)
    return tmptens

def res_conv_block_2016(data, n_filters, filter_size):
    tmptens = tf.layers.batch_normalization(data, training=is_training)
    tmptens = tf.nn.relu(tmptens)
    tmptens = tf.layers.conv2d(tmptens, n_filters, filter_size, padding="same")
    tmptens = tf.layers.batch_normalization(tmptens, training=is_training)
    tmptens = tf.nn.relu(tmptens)
    tmptens = tf.layers.conv2d(tmptens, n_filters, filter_size, padding="same")
    shortcut = tf.layers.batch_normalization(data, training=is_training)
    shortcut = tf.nn.relu(shortcut)
    shortcut = tf.layers.conv2d(shortcut, n_filters, filter_size, padding="same")
    tmptens = tf.add(shortcut, tmptens)
    return tmptens

def no_act_block_2015(data, n_filters, filter_size):
    tmptens = conv_norm_block(data, n_filters, filter_size)
    tmptens = tf.layers.conv2d(tmptens, n_filters, filter_size, padding="same")
    tmptens = tf.add(data, tmptens)
    return tmptens

def bn_after_add_block(data, n_filters, filter_size):
    tmptens = conv_norm_block(data, n_filters, filter_size)
    tmptens = tf.layers.conv2d(tmptens, n_filters, filter_size, padding="same")
    tmptens = tf.add(data, tmptens)
    tmptens = tf.layers.batch_normalization(data, training=is_training)
    return tf.nn.relu(tmptens)

def inception_v1_block(data, inc1_n_filters, inc3_n_filters, in5_n_filters, pool_n_filters, out_n_filters):
    tmptens = tf.layers.batch_normalization(data, training=is_training)
    tmptens = tf.nn.relu(tmptens)
    inc1net = tf.layers.conv2d(tmptens, inc1_n_filters, 1, padding="same")
    inc1net = tf.nn.relu(inc1net)
    inc3net = tf.layers.conv2d(tmptens, inc3_n_filters//2, 1, padding="same")
    inc3net = tf.nn.relu(inc3net)
    inc3net = tn.layers.conv2d(inc3net, inc3_n_filters, 3, padding="same")
    inc3net = tf.nn.relu(inc3net)
    inc5net = tf.layers.conv2d(tmptens, inc5_n_filters//2, 1, padding="same")
    inc5net = tf.nn.relu(inc5net)
    inc5net = tf.layers.conv2d(inc5net, inc5_n_filters, 5, padding="same")
    inc5net = tf.nn.relu(inc5net)
    poolnet = tf.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(tmptens)
    poolnet = tf.layers.conv2d(poolnet, pool_n_filters, 1, padding="same")
    poolnet = tf.nn.relu(poolnet)
    tmptens = tf.concat([tmptens, inc1net, inc3net, inc5net, poolnet], axis=3)
    tmptens = tf.layers.conv2d(tmptens, out_n_filters, 1, padding="same")
    tmptens = tf.add(data, tmptens)
    return tmptens

def EvANet(in_data):
    evanet = conv_norm_block(in_data, 16, 3)

    evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
    evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
    evanet = res_block_2016(evanet, 16, 3)
    evanet = inception_v1_block(evanet, 8, 16, 8, 32, 16)
    evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
    evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
    evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)

    evanet = tf.layers.conv2d(evanet, 32, 1, strides=2, padding="same")

    evanet = res_block_2015(evanet, 32, 3)
    evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
    evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
    evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
    evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
    evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
    evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)

    evanet = tf.layers.conv2d(evanet, 64, 1, strides=2, padding="same")

    evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
    evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
    evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
    evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
    evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
    evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
    evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)

    evanet = tf.layers.batch_normalization(evanet, training=is_training)
    evanet = tf.nn.relu(evanet)
    evanet = tf.layers.AveragePooling2D(pool_size=(3,3), strides=(1,1), padding="same")(evanet)

    net_out = tf.layers.dense(evanet, num_classes)
    _activation_summary(net_out)

    return net_out

# the following functions are based on code from:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# modified by Nils Kornfeld
#
#BEGIN APACHE LICENSED CODE
def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    num_batches_per_epoch = num_examples_per_epoch_for_train/batch_size
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

    lr = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, learning_rate_decay_factor, staircase=True)

#END APACHE LICENSED CODE



def train_evanet(X):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for iters in range(int(X.shape[0]//batch_size)):
                epoch_x = X[iters*batch_size:]
