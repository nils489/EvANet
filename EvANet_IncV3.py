from keras.datasets import cifar10
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Flatten
from keras.layers import Input
from keras.layers.merge import Add, Concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.utils import to_categorical


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

def inception_v3_block(in_blob, inc1_width, inc3_width, inc3_3_width,
                       pool_width, out_width):
    tmpnet = BatchNormalization()(in_blob)
    tmpnet = Activation('relu')(tmpnet)
    inc1net = Conv2D(inc1_width, 1, padding='same')(tmpnet)
    inctnet = Activation('relu')(inc1net)
    inc3net = Conv2D(inc3_width//2, 1, padding='same')(tmpnet)
    inc3net = Activation('relu')(inc3net)
    inc3net = Conv2D(inc3_width, 3, padding='same')(inc3net)
    inc3net = Activation('relu')(inc3net)
    inc3_3net = Conv2D(inc3_3_width//2, 1, padding='same')(tmpnet)
    inc3_3net = Activation('relu')(inc3_3net)
    inc3_3net = Conv2D(inc3_3_width, 3, padding='same')(inc3_3net)
    inc3_3net = Activation('relu')(inc3_3net)
    inc3_3net = Conv2D(inc3_3_width, 3, padding='same')(inc3_3net)
    inc3_3net = Activation('relu')(inc3_3net)
    poolnet = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(tmpnet)
    poolnet = Conv2D(pool_width, 1, padding='same')(poolnet)
    poolnet = Activation('relu')(poolnet)
    tmpnet = Concatenate(axis=3)([inc1net, inc3net, inc3_3net, poolnet])
    tmpnet = Conv2D(out_width, 1, padding='same')(tmpnet)
    tmpnet = Add()([in_blob, tmpnet])
    return tmpnet

num_classes = 10
num_epochs = 200
batch_size = 128

(X, Y), (test_x, test_y) = cifar10.load_data()

#X = X.reshape([-1,32,32,3])
#test_x = test_x.reshape([-1,32,32,3])
X = X.astype('float32')
text_x = test_x.astype('float32')
#X /= 255
#test_x /= 255

Y = to_categorical(Y, num_classes)
test_y = to_categorical(test_y, num_classes)

datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization = True,
                             width_shift_range=0.25, height_shift_range=0.25,
                             vertical_flip=True)

datagen.fit(X)
a = Input(shape=(32,32,3))

evanet = conv_norm_block(a, 16, 3)

evanet = inception_v3_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v3_block(evanet, 8, 16, 32, 8, 16)
evanet = res_block_2016(evanet, 16, 3)
evanet = inception_v3_block(evanet, 8, 16, 8, 32, 16)
evanet = inception_v3_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v3_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v3_block(evanet, 8, 16, 32, 8, 16)

evanet_conv = Conv2D(16, 1, strides=2, padding='same')(evanet)
evanet_pool = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                           padding='same')(evanet)
evanet = Concatenate(axis=3)([evanet_conv, evanet_pool])

evanet = res_block_2015(evanet, 32, 3)
evanet = inception_v3_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v3_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v3_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v3_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v3_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v3_block(evanet, 16, 32, 64, 16, 32)

evanet_conv = Conv2D(32, 1, strides=2, padding='same')(evanet)
evanet_pool = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                           padding='same')(evanet)
evanet = Concatenate(axis=3)([evanet_conv, evanet_pool])

evanet = inception_v3_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v3_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v3_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v3_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v3_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v3_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v3_block(evanet, 32, 64, 128, 32, 64)

evanet = BatchNormalization()(evanet)
evanet = Activation('relu')(evanet)
evanet = AveragePooling2D(pool_size=(3, 3), strides=(1,1),
                          padding='same')(evanet)

evanet = Flatten()(evanet)
evanet_out = Dense(num_classes, activation='softmax')(evanet)

model = Model(inputs=a, outputs=evanet_out)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
print("X.shape: ", X.shape)
print("Y.shape: ", Y.shape)
print("test_x.shape: ", test_x.shape)
print("test_y.shape: ", test_y.shape)
model.fit_generator(datagen.flow(X,Y, batch_size=batch_size),
                    steps_per_epoch=(len(X)//batch_size), epochs=num_epochs)
                    #validation_data=(test_x, test_y))
