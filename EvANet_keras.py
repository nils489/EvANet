import tensorflow as tf

def conv_norm_block(in_blob, width, filter_size):
    tmpnet = tf.keras.layers.Conv2D(width, filter_size,
                                    padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(in_blob)
    tmpnet = tf.keras.layers.BatchNormalization()(tmpnet)
    return tf.keras.layers.Activation('relu')(tmpnet)

def res_block_2015(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = tf.keras.layers.Conv2D(width, filter_size, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    tmpnet = tf.keras.layers.BatchNormalization()(tmpnet)
    tmpnet = tf.keras.layers.Add()([in_blob, tmpnet])
    tmpnet = tf.keras.layers.Activation('relu')(tmpnet)
    return tmpnet

def res_block_2016(in_blob, width, filter_size):
    tmpnet = tf.keras.layers.BatchNormalization()(in_blob)
    tmpnet = tf.keras.layers.Activation('relu')(tmpnet)
    tmpnet = tf.keras.layers.Conv2D(width, filter_size, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    tmpnet = tf.keras.layers.BatchNormalization()(tmpnet)
    tmpnet = tf.keras.layers.Activation('relu')(tmpnet)
    tmpnet = tf.keras.layers.Conv2D(width, filter_size, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    tmpnet = tf.keras.layers.Add()([in_blob, tmpnet])
    return tmpnet

def no_relu(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = tf.keras.layers.Conv2D(width, filter_size, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    tmpnet = tf.keras.layers.BatchNormalization()(tmpnet)
    tmpnet = tf.keras.layers.Add()([in_blob, tmpnet])
    return tmpnet

def res_conv_block_2016(in_blob, width, filter_size):
    tmpnet = tf.keras.layers.BatchNormalization()(in_blob)
    tmpnet = tf.keras.layers.Activation('relu')(tmpnet)
    tmpnet = tf.keras.layers.Conv2D(width, filter_size, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    tmpnet = tf.keras.layers.BatchNormalization()(tmpnet)
    tmpnet = tf.keras.layers.Activation('relu')(tmpnet)
    tmpnet = tf.keras.layers.Conv2D(widht, filter_size, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    shortcut = tf.keras.layers.BatchNormalization()(in_blob)
    shortcut = tf.keras.layers.Activation('relu')(shortcut)
    shortcut = tf.keras.layers.Conv2D(width, filter_size, padding='same',
                                      kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(shortcut)
    tmpnet = tf.keras.layers.Add()([shortcut, tmpnet])
    return tmpnet

def no_act_block_2015(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = tf.keras.layers.Conv2D(widht, filter_size, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    tmpnet = tf.keras.layers.Add()([in_blob, tmpnet])
    return tmpnet

def bn_after_add_block(in_blob, width, filter_size):
    tmpnet = conv_norm_block(in_blob, width, filter_size)
    tmpnet = tf.keras.layers.Conv2D(width, filter_size, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    tmpnet = tf.keras.layers.Add()([in_blob, tmpnet])
    tmpnet = tf.keras.layers.BatchNormalization()(tmpnet)
    tmpnet = tf.keras.layers.Activation('relu')(tmpnet)
    return tmpnet

def inception_v1_block(in_blob, inc1_width, inc3_width, inc5_width, pool_width,
                       out_width):
    tmpnet = tf.keras.layers.BatchNormalization()(in_blob)
    tmpnet = tf.keras.layers.Activation('relu')(tmpnet)
    inc1net = tf.keras.layers.Conv2D(inc1_width, 1, padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    inc1net = tf.keras.layers.Activation('relu')(inc1net)
    inc3net = tf.keras.layers.Conv2D(inc3_width//2, 1, padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    inc3net = tf.keras.layers.Activation('relu')(inc3net)
    inc3net = tf.keras.layers.Conv2D(inc3_width, 3, padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(inc3net)
    inc3net = tf.keras.layers.Activation('relu')(inc3net)
    inc5net = tf.keras.layers.Conv2D(inc5_width//2, 1, padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    inc5net = tf.keras.layers.Activation('relu')(inc5net)
    inc5net = tf.keras.layers.Conv2D(inc5_width, 5, padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(inc5net)
    inc5net = tf.keras.layers.Activation('relu')(inc5net)
    poolnet = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1,1),
                                           padding='same')(tmpnet)
    poolnet = tf.keras.layers.Conv2D(pool_width, 1, padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(poolnet)
    poolnet = tf.keras.layers.Activation('relu')(poolnet)
    tmpnet = tf.keras.layers.Concatenate(axis=3)([tmpnet, inc1net, inc3net,
                                                  inc5net, poolnet])
    tmpnet = tf.keras.layers.Conv2D(out_width, 1, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(tmpnet)
    tmpnet = tf.keras.layers.Add()([in_blob, tmpnet])
    return tmpnet

num_classes = 10
num_epochs = 500
batch_size = 128
weight_reg = 0.0001

(X, Y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

#X = X.reshape([-1,32,32,3])
#test_x = test_x.reshape([-1,32,32,3])
#X = X.astype('float32')
#text_x = test_x.astype('float32')
#X /= 255
#test_x /= 255

Y = tf.keras.utils.to_categorical(Y, num_classes)
test_y = tf.keras.utils.to_categorical(test_y, num_classes)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization = True,
                             width_shift_range=0.25, height_shift_range=0.25,
                             vertical_flip=True)

valgen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
                                                featurewise_std_normalization=True,
                                                width_shift_range=0,
                                                height_shift_range=0,
                                                vertical_flip=False)
valgen.fit(test_x)
datagen.fit(X)
a = tf.keras.Input(shape=(32,32,3))

evanet = conv_norm_block(a, 16, 3)

evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = res_block_2016(evanet, 16, 3)
evanet = inception_v1_block(evanet, 8, 16, 8, 32, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)
evanet = inception_v1_block(evanet, 8, 16, 32, 8, 16)

evanet = tf.keras.layers.Conv2D(32, 1, strides=2, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(evanet)

evanet = res_block_2015(evanet, 32, 3)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)
evanet = inception_v1_block(evanet, 16, 32, 64, 16, 32)

evanet = tf.keras.layers.Conv2D(64, 1, strides=2, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(evanet)

evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)
evanet = inception_v1_block(evanet, 32, 64, 128, 32, 64)

evanet = tf.keras.layers.BatchNormalization()(evanet)
evanet = tf.keras.layers.Activation('relu')(evanet)
evanet = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1,1),
                          padding='same')(evanet)

evanet = tf.keras.layers.Flatten()(evanet)
evanet_out = tf.keras.layers.Dense(num_classes, activation='softmax',
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_reg))(evanet)

sgd = tf.keras.optimizers.SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=False)

model = tf.keras.models.Model(inputs=a, outputs=evanet_out)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
print("X.shape: ", X.shape)
print("Y.shape: ", Y.shape)
print("test_x.shape: ", test_x.shape)
print("test_y.shape: ", test_y.shape)
model.fit_generator(datagen.flow(X,Y, batch_size=batch_size),
                    steps_per_epoch=(len(X)//batch_size), epochs=num_epochs,
                    validation_data=valgen.flow(test_x, test_y),
                                            validation_steps=len(test_y))
model.save("/PATH/TO/YOUR/MODELS/EvANet_model_"+num_epochs+"epochs.h5")
