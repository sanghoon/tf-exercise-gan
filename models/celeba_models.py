from utils import *


class DCGAN_D(object):
    # Define network hyper-parameters
    def __init__(self, n_out=10, last_act=tf.identity, **kwargs):
        self.name = 'dcgan_d'        # default name
        self.n_out = n_out
        self.k = 5
        self.last_act = last_act
        self.kwargs = kwargs

    # Network is divided into two different parts
    def former(self, x, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)  # Clone

        # Enable BN & Use leaky relu for all layers
        if 'bn' not in kwargs.keys():
            kwargs['bn'] = True
        if 'act' not in kwargs.keys():
            kwargs['act'] = leaky_relu

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            # TODO: Check the shape of x0

            conv1 = conv2d('conv1', x, [self.k, self.k, 03, 64], stride=2, **kwargs)            # 64 x 64 => 32
            conv2 = conv2d('conv2', conv1, [self.k, self.k, 64, 128], stride=2, **kwargs)       # 32 x 32 => 16
            conv3 = conv2d('conv3', conv2, [self.k, self.k, 128, 256], stride=2, **kwargs)      # 16 x 16 => 8
            conv4 = conv2d('conv4', conv3, [self.k, self.k, 256, 512], stride=2, **kwargs)      # 8 x 8 => 4

        return conv4

    def latter(self, conv3, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)  # Clone

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            kwargs['bn'] = False
            out = fc('fc1',
                   tf.reshape(conv3, [-1, 4 * 4 * 512]), [4 * 4 * 512, self.n_out],
                   self.last_act, **kwargs)                                         # No BN at the last layer

        return out

    # Instantiate the network
    def __call__(self, x, name=None, reuse=False, **kwargs):
        feat = self.former(x, name, reuse, **kwargs)
        out = self.latter(feat, name, reuse, **kwargs)

        return out


class DCGAN_G(object):
    # Define network hyper-parameters
    def __init__(self, n_in=128, last_act=tf.identity, **kwargs):
        self.name = 'dcgan_g'        # default name
        self.n_in = n_in
        self.k = 5
        self.last_act = last_act
        self.kwargs = kwargs

    # Network is divided into two different parts (so that we can experiment networks with *shared* former layers)
    def former(self, z, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)      # Clone

        # Enable BN & Use ELU for all former layers
        if 'bn' not in kwargs.keys():
            kwargs['bn'] = True
        if 'act' not in kwargs.keys():
            kwargs['act'] = tf.nn.elu

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            h = fc('fc1', z, [self.n_in, 4 * 4 * 1024], **kwargs)
            h = tf.reshape(h, [-1, 4, 4, 1024])
            h = deconv2d('deconv1', h, [-1, 8, 8, 512], [self.k, self.k, 512, 1024], stride=2, **kwargs)
            h = deconv2d('deconv2', h, [-1, 16, 16, 256], [self.k, self.k, 256, 512], stride=2, **kwargs)
            h = deconv2d('deconv3', h, [-1, 32, 32, 128], [self.k, self.k, 128, 256], stride=2, **kwargs)

        return h

    def latter(self, h, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)  # Clone

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            # No BN, No activation
            kwargs['bn'] = False
            kwargs['act'] = tf.identity
            h = deconv2d('deconv4', h, [-1, 64, 64, 3], [self.k, self.k, 3, 128], stride=2, **kwargs)

            # Apply additional activation func if specified
            h = self.last_act(h, 'out')

        return h

    # Instantiate the network
    def __call__(self, x, name=None, reuse=False, **kwargs):
        feat = self.former(x, name, reuse, **kwargs)
        out = self.latter(feat, name, reuse, **kwargs)

        return out