from utils import *


class SimpleCNN(object):
    # Define network hyper-parameters
    def __init__(self, n_out=10, last_act=tf.identity, **kwargs):
        self.name = 'simplecnn'        # default name
        self.n_out = n_out
        self.k = 5
        self.last_act = last_act
        self.kwargs = kwargs

    # Network is divided into two different parts
    def former(self, x, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)  # Clone

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            # TODO: Check the shape of x0

            conv1 = conv2d('conv1_1', x, [self.k, self.k, 01, 32], stride=2, **kwargs)        # 28 x 28 => 14
            conv2 = conv2d('conv2_1', conv1, [self.k, self.k, 32, 64], stride=2, **kwargs)    # 14 x 14 => 7

        return conv2

    def latter(self, conv3, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)  # Clone

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            kwargs['bn'] = False
            out = fc('fc1',
                   tf.reshape(conv3, [-1, 7 * 7 * 64]), [7 * 7 * 64, self.n_out],
                   self.last_act, **kwargs)                                         # No BN at the last layer

        return out

    # Instantiate the network
    def __call__(self, x, name=None, reuse=False, **kwargs):
        feat = self.former(x, name, reuse, **kwargs)
        out = self.latter(feat, name, reuse, **kwargs)

        return out


class SimpleGEN(object):
    # Define network hyper-parameters
    def __init__(self, n_in=128, last_act=tf.identity, **kwargs):
        self.name = 'simplegen'        # default name
        self.n_in = n_in
        self.k = 5
        self.last_act = last_act
        self.kwargs = kwargs

    # Network is divided into two different parts (so that we can experiment networks with *shared* former layers)
    def former(self, z, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)      # Clone

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            h = fc('fc1', z, [self.n_in, 7 * 7 * 64], act=tf.nn.elu, **kwargs)
            h = tf.reshape(h, [-1, 7, 7, 64])  # 7 x 7
            h = deconv2d('deconv1', h, [-1, 14, 14, 32], [self.k, self.k, 32, 64], stride=2, act=tf.nn.elu, **kwargs)

        return h

    def latter(self, h, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)  # Clone

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            kwargs['bn'] = False  # Last layer only
            kwargs['act'] = tf.identity
            h = deconv2d('deconv2', h, [-1, 28, 28, 01], [self.k, self.k, 01, 32], stride=2, **kwargs)
            h = self.last_act(h, 'out')

        return h

    # Instantiate the network
    def __call__(self, x, name=None, reuse=False, **kwargs):
        feat = self.former(x, name, reuse, **kwargs)
        out = self.latter(feat, name, reuse, **kwargs)

        return out