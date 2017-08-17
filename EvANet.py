import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import input_data, fully_connected, activation
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.merge_ops  import merge
from tflearn.optimizers import SGD
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import DataPreprocessing
import tflearn.datasets.cifar10 as cifar10

def conv_norm_block(in_blob, width, filter_size):
    tmpnet = conv_2d(in_blob, width, filter_size)
    tmpnet = batch_normalization(tmpnet)
    tmpnet = activation(tmpnet, activation='relu')
    return tmpnet

def res_block_2015(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = conv_2d(tmpnet, width, filter_size)
    tmpnet = batch_normalization(tmpnet)
    tmpnet = merge(tensors_list=[in_blob,tmpnet], mode='elemwise_sum')
    tmpnet = activation(tmpnet, activation='relu')
    return tmpnet

def res_block_2016(in_blob, width, filter_size):
    tmpnet = batch_normalization(in_blob)
    tmpnet = activation(tmpnet, activation='relu')
    tmpnet = conv_2d(tmpnet, width, filter_size)
    tmpnet = batch_normalization(tmpnet)
    tmpnet = activation(tmpnet, activation='relu')
    tmpnet = conv_2d(tmpnet, width, filter_size)
    tmpnet = merge(tensors_list=[in_blob,tmpnet], mode='elemwise_sum')
    return tmpnet

def no_relu(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = conv_2d(tmpnet, width, filter_size)
    tmpnet = batch_normalization(tmpnet)
    tmpnet = merge(tensors_list=[in_blob,tmpnet], mode='elemwise_sum')
    return tmpnet

def res_conv_block_2016(in_blob, width, filter_size):
    tmpnet = batch_normalization(in_blob)
    tmpnet = activation(tmpnet, activation='relu')
    tmpnet = conv_2d(tmpnet, width, filter_size)
    tmpnet = batch_normalization(tmpnet)
    tmpnet = activation(tmpnet, activation='relu')
    tmpnet = conv_2d(tmpnet, width, filter_size)
    shortcut = batch_normalization(in_blob)
    shortcut = activation(shortcut, activation='relu')
    shortcut = conv_2d(shortcut, width, filter_size)
    tmpnet = merge(tensors_list=[shortcut,tmpnet], mode='elemwise_sum')
    return tmpnet

def no_act_block_2015(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = conv_2d(tmpnet, width, filter_size)
    tmpnet = merge(tensors_list=[in_blob,tmpnet], mode='elemwise_sum')
    return tmpnet

def bn_after_add_block(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = conv_2d(tmpnet, width, filter_size)
    tmpnet = merge(tensors_list=[in_blob,tmpnet], mode='elemwise_sum')
    tmpnet = batch_normalization(tmpnet)
    tmpnet = activation(tmpnet, activation='relu')
    return tmpnet

def inception_v1_block(in_blob, inc1_width, inc3_width, inc5_width, pool_width,
                       out_width):
    tmpnet = batch_normalization(in_blob)
    tmpnet = activation(tmpnet, activation='relu')
    inc1net = conv_2d(tmpnet, inc1_width, 1)
    inc1net = activation(inc1net, activation='relu')
    inc3net = conv_2d(tmpnet, inc3_width/2, 1)
    inc3net = activation(inc3net, activation='relu')
    inc3net = conv_2d(inc3net, inc3_width, 3)
    inc3net = activation(inc3net, activation='relu')
    inc5net = conv_2d(tmpnet, inc5_width/2, 1)
    inc5net = activation(inc5net, activation='relu')
    inc5net = conv_2d(inc5net, inc5_width, 3)
    inc5net = activation(inc5net, activation='relu')
    poolnet = max_pool_2d(tmpnet, 3, strides=1)
    poolnet = conv_2d(poolnet, pool_width, 1)
    poolnet = activation(poolnet, activation='relu')
    tmpnet = merge(tensors_list=[inc1net,inc3net,inc5net,poolnet],
                   mode='concat', axis=3)
    tmpnet = conv_2d(tmpnet, out_width, 1)
    tmpnet = merge(tensors_list=[in_blob,tmpnet], mode='elemwise_sum')
    return tmpnet


(X, Y), (test_x, test_y) = cifar10.load_data(one_hot=True)

X = X.reshape([-1,32,32,3])
test_x = test_x.reshape([-1,32,32,3])

aug = ImageAugmentation()
aug.add_random_crop((32,32),6)
aug.add_random_flip_leftright()

prec = DataPreprocessing()
prec.add_featurewise_stdnorm()
prec.add_featurewise_zero_center()

evanet = input_data(shape=[None,32,32,3], data_preprocessing=prec, data_augmentation=aug, name='input')

evanet = conv_norm_block(evanet, 16, 3)

evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = res_block_2016(evanet, 16, 3)
evanet = inception_v1_block(evanet, 8, 16 , 8, 32, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)

evanet = conv_2d(evanet, 32, 1, strides=2, trainable='false')

evanet = res_block_2015(evanet, 32, 3)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)

evanet = conv_2d(evanet, 64, 1, strides=2, trainable='false')

evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)

evanet = batch_normalization(evanet)
evanet = activation(evanet, activation='relu')
evanet = avg_pool_2d(evanet, 3)
evanet = fully_connected(evanet, 10, activation='softmax')

sgd = SGD(learning_rate=0.1, lr_decay=0.1, decay_step=2500000)
evanet = regression(evanet, optimizer=sgd, loss='categorical_crossentropy',
                    name='targets')

model = tflearn.DNN(evanet)

model.fit({'input':X},{'targets':Y}, n_epoch=200,
          validation_set=({'input':test_x},{'targets':test_y}),
          snapshot_step=500, batch_size=128, show_metric=True, run_id='cifar10')

model.save('tflearn_evanet.model')
