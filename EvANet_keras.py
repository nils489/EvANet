from keras.datasets import cifar10
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense
from keras.layers import Input
from keras.layers.merge import Add, Concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model


def conv_norm_block(in_blob, width, filter_size):
    tmpnet = Conv2D(width, filter_size, padding='same')(in_blob)
    tmpnet = BatchNormalization()(tmpnet)
    tmpnet = Activation('relu')(tmpnet)
    return tmpnet

def res_block_2015(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = Conv2D(width, filter_size, padding='same')(tmpnet)
    tmpnet = BatchNormalization()(tmpnet)
    tmpnet = Add()([in_blob, tmpnet])
    tmpnet = Activation('relu')(tmpnet)
    return tmpnet

def res_block_2016(in_blob, width, filter_size):
    tmpnet = BatchNormalization()(in_blob)
    tmpnet = Activation('relu')(tmpnet)
    tmpnet = Conv2D(width, filter_size, padding='same')(tmpnet)
    tmpnet = BatchNormalization()(tmpnet)
    tmpnet = Activation('relu')(tmpnet)
    tmpnet = Conv2D(width, filter_size, padding='same')(tmpnet)
    tmpnet = Add()([in_blob, tmpnet])
    return tmpnet

def no_relu(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = Conv2D(width, filter_size, padding='same')(tmpnet)
    tmpnet = BatchNormalization()(tmpnet)
    tmpnet = Add()([in_blob, tmpnet])
    return tmpnet

def res_conv_block_2016(in_blob, width, filter_size):
    tmpnet = BatchNormalization()(in_blob)
    tmpnet = Activation('relu')(tmpnet)
    tmpnet = Conv2D(width, filter_size, padding='same')(tmpnet)
    tmpnet = BatchNormalization()(tmpnet)
    tmpnet = Activation('relu')(tmpnet)
    tmpnet = Conv2D(widht, filter_size, padding='same')(tmpnet)
    shortcut = BatchNormalization()(in_blob)
    shortcut = Activation('relu')(shortcut)
    shortcut = Conv2D(width, filter_size, padding='same')(shortcut)
    tmpnet = Add()([shortcut, tmpnet])
    return tmpnet

def no_act_block_2015(in_blob, widht, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = Conv2D(widht, filter_size, padding='same')(tmpnet)
    tmpnet = Add()([in_blob, tmpnet])
    return tmpnet

def bn_after_add_block(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = Conv2D(width, filter_size, padding='same')(tmpnet)
    tmpnet = Add()([in_blob, tmpnet])
    tmpnet = BatchNormalization()(tmpnet)
    tmpnet = Activation('relu')(tmpnet)
    return tmpnet

def inception_v1_block(in_blob, inc1_width, inc3_width, inc5_width, pool_width,
                       out_width):
    tmpnet = BatchNormalization()(in_blob)
    tmpnet = Activation('relu')(tmpnet)
    inc1net = Conv2D(inc1_width, 1, padding='same')(tmpnet)
    inc1net = Activation('relu')(inc1net)
    inc3net = Conv2D(inc3_width//2, 1, padding='same')(tmpnet)
    inc3net = Activation('relu')(inc3net)
    inc3net = Conv2D(inc3_width, 3, padding='same')(inc3net)
    inc3net = Activation('relu')(inc3net)
    inc5net = Conv2D(inc5_width//2, 1, padding='same')(tmpnet)
    inc5net = Activation('relu')(inc5net)
    inc5net = Conv2D(inc5_width, 5, padding='same')(inc5net)
    inc5net = Activation('relu')(inc5net)
    poolnet = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(tmpnet)
    poolnet = Conv2D(pool_width, 1, padding='same')(poolnet)
    poolnet = Activation('relu')(poolnet)
    tmpnet = Concatenate(axis=3)([inc1net, inc3net, inc5net, poolnet])
    tmpnet = Conv2D(out_width, 1, padding='same')(tmpnet)
    tmpnet = Add()([in_blob, tmpnet])
    return tmpnet

(X, Y), (test_x, test_y) = cifar10.load_data()

#X = X.reshape([-1,32,32,3])
#test_x = test_x.reshape([-1,32,32,3])

datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization = True,
                             width_shift_range=0.25, height_shift_range=0.25,
                             vertical_flip=True)

datagen.fit(X)
a = Input(shape=(32,32,3))

evanet = conv_norm_block(a, 16, 3)

evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = res_block_2016(evanet, 16, 3)
evanet = inception_v1_block(evanet, 8, 16, 8, 32, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)

evanet = Conv2D(32, 1, strides=2, padding='same')(evanet)

evanet = res_block_2015(evanet, 32, 3)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)

evanet = Conv2D(64, 1, strides=2, padding='same')(evanet)

evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)

evanet = BatchNormalization()(evanet)
evanet = Activation('relu')(evanet)
evanet = AveragePooling2D(pool_size=(3, 3), strides=(1,1),
                          padding='same')(evanet)
evanet_out = Dense(10, activation='softmax')(evanet)

model = Model(inputs=a, outputs=evanet_out)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
print(X.shape)
model.fit_generator(datagen.flow(X,Y, batch_size=128),
                    steps_per_epoch=(len(X)/128), epochs=200,
                    validation_data=(test_x, test_y))
