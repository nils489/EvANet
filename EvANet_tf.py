import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


num_classes = 10
num_epochs  = 50
batch_size  = 128

(X, Y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
X = X.astype('float32')
test_x = test_x.astype('float32')

Y = tf.contrib.keras.utils.to_categorical(Y, num_classes)
test_y = tf.contrib.keras.utils.to_categorical(test_y, num_classes)

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
