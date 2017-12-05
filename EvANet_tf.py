import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


num_classes = 10
num_epochs  = 50
batch_size  = 128

(X, Y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
X = X.astype('float32')
test_x = test_x.astype('float32')

Y = tf.keras.utils.to_categorical(Y, num_classes)
test_y = tf.keras.utils.to_categorical(test_y, num_classes)

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

def evanet(data, n_filters, filter_size):
    tmptens = conv_norm_block(data, n_filters, filter_size)

    tmptens = tf.layers.average_pooling2d(tmptens, pool_size(3,3),
                                          strides=(1,1), padding="same")
    net_out = tf.layers.dense(tmptens, num_classes)
    return net_out

def train_evanet(X):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for iters in range(int(X.shape[0]//batch_size)):
                epoch_x = X[iters*batch_size:]



print("X.shape: ", X.shape)
print("Y.shape: ", Y.shape)
print("test_x.shape: ", test_x.shape)
print("test_y.shape: ", test_y.shape)
