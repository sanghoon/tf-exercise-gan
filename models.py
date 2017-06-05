from utils import *

def xavier_init(size):
    _in_dim = size[0]
    _stddev = 1. / tf.sqrt(_in_dim / 2.)
    return tf.random_normal(shape=size, stddev=_stddev)

DIM_Z = 32
DIM_X = 28 * 28

# Simple generator
def simple_gen(z, name='simple_gen', dim_h=128, **kwargs):
    with tf.variable_scope(name):
        l1 = fc('fc1', z, [DIM_Z, dim_h], **kwargs)
        l2 = fc('fc2', l1, [dim_h, dim_h], **kwargs)
        l3 = fc('fc3', l2, [dim_h, DIM_X], act=tf.nn.sigmoid, **kwargs)
        l3 = tf.reshape(l3, [-1, 28, 28])

    return l3

# Simple discriminator
def simple_net(x, name='simple_net', n_out=10, dim_h=128, **kwargs):
    with tf.variable_scope(name):
        l0 = tf.reshape(x, [-1, DIM_X])
        l1 = fc('fc1', l0, [DIM_X, dim_h], **kwargs)
        l2 = fc('fc2', l1, [dim_h, dim_h], **kwargs)
        if 'bn' in kwargs:
            del(kwargs['bn'])
        l3 = fc('fc3', l2, [dim_h, n_out], act=tf.identity, bn=False, **kwargs)

        # How about solving 2-class classification?
        # What if we add a sigmoid function here?

    return l3


def simple_cnn(x1, name='CNN', n_out=10, is_training=True, reuse=False, do_rate=0.5):
    kwargs = {'is_training': is_training, 'reuse': reuse}

    with tf.variable_scope(name):
        # Check the shape of x0
        l1 = conv2d('conv1_1', x1, [3, 3, 1, 16], **kwargs)                  # 28 x 28
        l2 = conv2d('conv2_1', l1, [3, 3, 16, 32], stride=2, **kwargs)       # 14 x 14
        l3 = conv2d('conv3_1', l2, [3, 3, 32, 64], stride=2, **kwargs)       # 7 x 7
        l4 = fc('fc1',
                tf.reshape(l3, [-1, 7 * 7 * 64]), [7 * 7 * 64, 256], **kwargs)
        l4 = tf.layers.dropout(l4, rate=do_rate, training=is_training)
        l5 = fc('fc2', l4, [256, n_out], tf.identity, bn=False, **kwargs)

    return l5

def simple_cnn_gen(z0, name='CNN_gen', is_training=True, reuse=False):
    kwargs = {'is_training': is_training, 'reuse': reuse}

    with tf.variable_scope(name):
        l1 = fc('fc1', z0, [32, 256], **kwargs)
        l1 = tf.reshape(l1, [-1, 7, 7, 64])                                         # 7 x 7
        l2 = deconv2d('deconv1', l1, [5, 5, 32, 64], stride=2, **kwargs)            # 14 x 14
        l3 = deconv2d('deconv2', l2, [5, 5, 16, 32], stride=2, **kwargs)            # 28 x 28
        l4 = deconv2d('deconv3', l3, [5, 5, 1, 16], stride=2, bn=False, **kwargs)   # 28 x 28
        l5 = tf.sigmoid(l4, 'out')

    return l5


