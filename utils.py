import tensorflow as tf


def get_trainable_params(scope_name):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)


def ops_copy_vars(src_scope, dst_scope, exclude_keys=['RMSProp']):
    src_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=src_scope)
    dst_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=dst_scope)

    src_dict = {}
    for i in xrange(len(src_vars)):
        key = src_vars[i].name
        key = key[(len(src_scope)):]  # Remove scope name
        src_dict[key] = src_vars[i]

    ops_list = []
    for i in xrange(len(dst_vars)):
        key = dst_vars[i].name
        key = key[(len(dst_scope)):]  # Remove scope name

        is_ignored = any(map(lambda x: x in key, exclude_keys))

        if is_ignored:
            continue

        # TODO: Error handling
        ops_list.append([dst_vars[i].assign(src_dict[key])])

    # XXX: Test on BN params

    return ops_list

def conv2d(name, in_var, shape, stride=1, act=tf.nn.relu, bn=False,
           is_training=True, reuse=False):
    # ordering: N W H C
    with tf.variable_scope(name, reuse=reuse) as scope:
        w_shape = shape        # filterH, filterW, inChns, outChns
        b_shape = shape[3:4]   # outChns

        w = tf.get_variable('w', w_shape,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        b = tf.get_variable('b', b_shape,
                            initializer=tf.constant_initializer(0.1))

        h = tf.nn.conv2d(in_var, w, strides=[1, stride, stride, 1], padding='SAME')
        h = tf.nn.bias_add(h, b)

        if bn:
            h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=is_training,
                                             scope='bn', reuse=reuse)

        h = act(h, name='out')

    return h


def deconv2d(name, in_var, shape, stride=1, act=tf.nn.relu, bn=False,
             is_training=True, reuse=False):
    # ordering: N W H C
    with tf.variable_scope(name, reuse=reuse) as scope:
        w_shape = shape         # filterH, filterW, outChns, inChns
        b_shape = shape[2:3]    # outChns

        # FIXME: Initialization of deconv
        w = tf.get_variable('w', w_shape,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', b_shape,
                            initializer=tf.constant_initializer(0.1))

        h = tf.nn.conv2d_transpose(in_var, w, strides=[1, stride, stride, 1], padding='SAME')
        h = tf.nn.bias_add(h, b)

        if bn:
            h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=is_training,
                                             scope='bn', reuse=reuse)

        h = act(h, name='out')

    return h


def fc(name, in_var, shape, act=tf.nn.relu, bn=False,
       is_training=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        w_shape = shape
        b_shape = shape[1:2]

        w = tf.get_variable('w', w_shape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable('b', b_shape,
                            initializer=tf.constant_initializer(0))

        h = tf.matmul(in_var, w) + b

        if bn:
            h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=is_training,
                                             scope='bn', reuse=reuse)

        h = act(h, name='out')

    return h

