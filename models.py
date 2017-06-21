from utils import *

def xavier_init(size):
    _in_dim = size[0]
    _stddev = 1. / tf.sqrt(_in_dim / 2.)
    return tf.random_normal(shape=size, stddev=_stddev)

DIM_Z = 128
DIM_X = 28 * 28

# Simple generator
def simple_gen(z, name='simple_gen', n_in=DIM_Z, dim_h=256, **kwargs):
    with tf.variable_scope(name):
        l1 = fc('fc1', z, [n_in, dim_h], **kwargs)
        l2 = fc('fc2', l1, [dim_h, dim_h], **kwargs)

        kwargs['bn'] = False
        l3 = fc('fc3', l2, [dim_h, DIM_X], act=tf.nn.sigmoid, **kwargs)
        l3 = tf.reshape(l3, [-1, 28, 28, 1])

    return l3

# Simple discriminator
def simple_net(x, name='simple_net', n_out=10, dim_h=256, **kwargs):
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


# Simple ConvNet
def simple_cnn(x, name='CNN', n_out=10, is_training=True, k=3,
               last_act=tf.identity, **kwargs):
    with tf.variable_scope(name):
        # TODO: Check the shape of x0
        h = x
        h = conv2d('conv1_1', h, [k, k, 01, 16], stride=1, **kwargs)        # 28 x 28
        h = conv2d('conv2_1', h, [k, k, 16, 32], stride=2, **kwargs)        # 14 x 14
        h = conv2d('conv3_1', h, [k, k, 32, 64], stride=2, **kwargs)        # 7 x 7

        kwargs['bn'] = False
        h = fc('fc1',
                tf.reshape(h, [-1, 7 * 7 * 64]), [7 * 7 * 64, n_out],
                last_act, **kwargs)

    return h

def simple_cnn_gen(z, name='CNN_gen', n_in=DIM_Z, k=3,
                   last_act=tf.sigmoid, **kwargs):
    # XXX: Is there a way to define conv2d_transpose without specifying the batchsize?
    with tf.variable_scope(name):
        h = fc('fc1', z, [n_in, 7 * 7 * 64], act=tf.nn.elu, **kwargs)
        h = tf.reshape(h, [-1, 7, 7, 64])                                                     # 7 x 7
        h = deconv2d('deconv1', h, [-1, 14, 14, 32], [k, k, 32, 64], stride=2, act=tf.nn.elu, **kwargs)
        h = deconv2d('deconv2', h, [-1, 28, 28, 16], [k, k, 16, 32], stride=2, act=tf.nn.elu, **kwargs)

        kwargs['bn'] = False  # Last layer only
        kwargs['act'] = tf.identity
        h = deconv2d('deconv3', h, [-1, 28, 28, 01], [k, k, 01, 16], stride=1, **kwargs)
        h = last_act(h, 'out')

    return h


# Deeper ConvNet
def deeper_cnn(x, name='CNN', n_out=10, is_training=True, k=3,
               last_act=tf.identity, **kwargs):
    with tf.variable_scope(name):
        # Check the shape of x0
        h = conv2d('conv1_1', x, [k, k, 1, 16], stride=1, **kwargs)         # 28 x 28
        h = conv2d('conv2_1', h, [k, k, 16, 32], stride=2, **kwargs)        # 14 x 14
        h = conv2d('conv2_2', h, [k, k, 32, 32], stride=1, **kwargs)        # 14 x 14
        h = conv2d('conv3_1', h, [k, k, 32, 64], stride=2, **kwargs)        # 7 x 7
        h = conv2d('conv3_2', h, [k, k, 64, 64], stride=1, **kwargs)        # 7 x 7

        kwargs['bn'] = False
        h = fc('fc1',
                tf.reshape(h, [-1, 7 * 7 * 64]), [7 * 7 * 64, n_out],
                last_act, **kwargs)

    return h

def deeper_cnn_gen(z, name='CNN_gen', n_in=DIM_Z, k=3,
                   last_act=tf.sigmoid, **kwargs):
    # XXX: Is there a way to define conv2d_transpose without specifying the batchsize?
    with tf.variable_scope(name):
        h = fc('fc1', z, [n_in, 7 * 7 * 64], **kwargs)
        h = tf.reshape(h, [-1, 7, 7, 64])                                                     # 7 x 7
        h = deconv2d('deconv1', h, [-1, 07, 07, 64], [k, k, 64, 64], stride=1, act=tf.nn.elu, **kwargs)
        h = deconv2d('deconv2', h, [-1, 14, 14, 32], [k, k, 32, 64], stride=2, act=tf.nn.elu, **kwargs)
        h = deconv2d('deconv3', h, [-1, 14, 14, 32], [k, k, 32, 32], stride=1, act=tf.nn.elu, **kwargs)
        h = deconv2d('deconv4', h, [-1, 28, 28, 16], [k, k, 16, 32], stride=2, act=tf.nn.elu, **kwargs)

        kwargs['bn'] = False  # Last layer only
        kwargs['act'] = tf.identity
        h = deconv2d('deconv5', h, [-1, 28, 28, 01], [k, k, 01, 16], stride=1, **kwargs)
        h = last_act(h, 'out')

    return h


models = {
    'simple_cnn': (simple_cnn_gen, simple_cnn),
    'deeper_cnn': (deeper_cnn_gen, deeper_cnn),
    'simple_fc': (simple_gen, simple_net),
}