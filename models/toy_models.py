from utils import *


class ToyNet(object):
    # Define network hyper-parameters
    def __init__(self, n_out=10, n_in=2, dim_h=256, last_act=tf.identity, **kwargs):
        self.name = 'toydisc'        # default name
        self.n_out = n_out
        self.last_act = last_act
        self.dim_x = n_in
        self.dim_h = dim_h
        self.kwargs = kwargs

    # Network is divided into two different parts
    def former(self, x, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)  # Clone

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            x = tf.reshape(x, [-1, self.dim_x])             # Reshape (if needed)

            fc1 = fc('fc1', x, [self.dim_x, self.dim_h], **kwargs)
            fc2 = fc('fc2', fc1, [self.dim_h, self.dim_h], **kwargs)

        return fc2

    def latter(self, fc2, name=None, reuse=False):
        if not name:
            name = self.name
        kwargs = dict(self.kwargs)  # Clone

        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()

            kwargs['bn'] = False        # No BN at the last layer
            kwargs['act'] = self.last_act
            out = fc('fc3', fc2, [self.dim_h, self.n_out], **kwargs)

        return out

    # Instantiate the network
    def __call__(self, x, name=None, reuse=False, **kwargs):
        feat = self.former(x, name, reuse, **kwargs)
        out = self.latter(feat, name, reuse, **kwargs)

        return out

