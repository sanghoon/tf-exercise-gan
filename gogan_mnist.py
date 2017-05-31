#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import os
# if DISPLAY is not defined
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')       # Use a different backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

DIM_Z = 16
DIM_X = 28 * 28
W_CLIP = 0.01
BATCH_SIZE = 100


# Load dataset
data = input_data.read_data_sets('data/mnist/', one_hot=True)

trainimg = data.train.images
testimg = data.test.images



def plot(samples, figId=None):
    if figId is None:
        fig = plt.figure(figsize=(4, 4))
    else:
        fig = plt.figure(figId, figsize=(4,4))

    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def xavier_init(size):
    _in_dim = size[0]
    _stddev = 1. / tf.sqrt(_in_dim / 2.)
    return tf.random_normal(shape=size, stddev=_stddev)

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# Simple generator
def simple_G(z, params=None, dim_h=128):
    if params is not None:
        raise NotImplementedError

    G_FC1_W = tf.Variable(xavier_init([DIM_Z, dim_h]))
    G_FC1_B = tf.Variable(tf.zeros(shape=[dim_h]))

    G_FC2_W = tf.Variable(xavier_init([dim_h, dim_h]))
    G_FC2_B = tf.Variable(tf.zeros(shape=[dim_h]))

    G_FC3_W = tf.Variable(xavier_init([dim_h, DIM_X]))
    G_FC3_B = tf.Variable(tf.zeros(shape=[DIM_X]))

    theta_G = [G_FC1_W, G_FC1_B, G_FC2_W, G_FC2_B, G_FC3_W, G_FC3_B]

    G_h1 = tf.nn.relu(tf.matmul(z, G_FC1_W) + G_FC1_B)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_FC2_W) + G_FC2_B)
    G_h3 = tf.nn.sigmoid(tf.matmul(G_h2, G_FC3_W) + G_FC3_B)

    return G_h3, theta_G

# Simple discriminator
def simple_D(x, params=None, dim_h=128):
    if params is None:
        D_FC1_W = tf.Variable(xavier_init([DIM_X, dim_h]))
        D_FC1_B = tf.Variable(tf.zeros(shape=[dim_h]))

        D_FC2_W = tf.Variable(xavier_init([dim_h, dim_h]))
        D_FC2_B = tf.Variable(tf.zeros(shape=[dim_h]))

        D_FC3_W = tf.Variable(xavier_init([dim_h, 1]))  # How about solving 2-class classification?
        D_FC3_B = tf.Variable(tf.zeros(shape=[1]))

        theta_D = [D_FC1_W, D_FC1_B, D_FC2_W, D_FC2_B, D_FC3_W, D_FC3_B]
    else:
        theta_D = params
        D_FC1_W = theta_D[0]
        D_FC1_B = theta_D[1]
        D_FC2_W = theta_D[2]
        D_FC2_B = theta_D[3]
        D_FC3_W = theta_D[4]
        D_FC3_B = theta_D[5]

    D_h1 = tf.nn.relu(tf.matmul(x, D_FC1_W) + D_FC1_B)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_FC2_W) + D_FC2_B)
    D_h3 = tf.matmul(D_h2, D_FC3_W) + D_FC3_B           # What if we add a sigmoid function here?

    return D_h3, theta_D


### WGAN
# Instantiate network
z0 = tf.placeholder(tf.float32, shape=[None, DIM_Z])
x0 = tf.placeholder(tf.float32, shape=[None, DIM_X])

G, theta_G = simple_G(z0)
D_real, theta_D = simple_D(x0)
D_fake, _ = simple_D(G, theta_D)

# Loss function for WGAN
D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4))  \
            .minimize(-D_loss, var_list=theta_D)
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4))  \
            .minimize(G_loss, var_list=theta_G)

clip_D = [p.assign(tf.clip_by_value(p, -W_CLIP, W_CLIP)) for p in theta_D]


### GoGAN
epsilon = 0.5   # Margin
l1 = 1.0        # Disc. loss
l2 = 0.5        # Rank. loss

G1, theta_G1 = simple_G(z0)
D1_real, theta_D1 = simple_D(x0)
D1_fake, theta_D1 = simple_D(G1, theta_D1)

G2, theta_G2 = simple_G(z0)
D2_real, theta_D2 = simple_D(x0)
D2_fake, theta_D2 = simple_D(G2, theta_D2)

D1_loss = tf.reduce_mean(tf.nn.relu(D1_fake + epsilon - D1_real))
G1_loss = -tf.reduce_mean(D1_fake)

D2_loss = tf.reduce_mean(tf.nn.relu(D2_fake + epsilon - D2_real)) * l1 \
        + tf.reduce_mean(tf.nn.relu(D1_fake + 2 * epsilon - D2_real)) * l2
G2_loss = -tf.reduce_mean(D2_fake)

clip_D1 = [p.assign(tf.clip_by_value(p, -W_CLIP, W_CLIP)) for p in theta_D1]
clip_D2 = [p.assign(tf.clip_by_value(p, -W_CLIP, W_CLIP)) for p in theta_D2]

G1_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)).minimize(G1_loss, var_list=theta_G1)
D1_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)).minimize(D1_loss, var_list=theta_D1)
G2_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)).minimize(G2_loss, var_list=theta_G2)
D2_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)).minimize(D2_loss, var_list=theta_D2)


# Session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))     # XXX: Is this necessary?
sess.run(tf.global_variables_initializer())

# Initial setup for visualization
outputs = [G, G1, G2]
figs = [None] * len(outputs)
fig_names = ['fig_WGAN_gen_{:04d}.png', 'fig_GGAN_1st_{:04d}.png', 'fig_GGAN_2nd_{:04d}.png']

if not os.path.exists('out/'):
    os.makedirs('out/')

print('{:>10}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}') \
    .format('Iters', 'WGAN_D', 'WGAN_G', 'GGAN_D1', 'GGAN_G1', 'GGAN_D2', 'GGAN_G2')

# 500 iterations = 1 epoch
N_ITERS = 500 * 1000
N_STAGE1 = 500 * 500
for it in range(N_ITERS):
    if it == N_STAGE1:
        # Copy vars. and start phase-2
        for var_from, var_to in zip(theta_D1, theta_D2):
            sess.run(var_to.assign(var_from))
        for var_from, var_to in zip(theta_G1, theta_G2):
            sess.run(var_to.assign(var_from))

        pass

    # Train WGAN
    for _ in range(5):
        batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)

        _, loss_WGAN_D, _ = sess.run(
            [D_solver, D_loss, clip_D],
            feed_dict={x0: batch_xs, z0: sample_z(BATCH_SIZE, DIM_Z)}
        )

    _, loss_WGAN_G = sess.run(
        [G_solver, G_loss],
        feed_dict={z0: sample_z(BATCH_SIZE, DIM_Z)}
    )

    # Train 1st-stage GoGAN
    for _ in range(5):
        _, loss_GGAN_D1, _ = sess.run(
            [D1_solver, D1_loss, clip_D1],
            feed_dict={x0: batch_xs, z0: sample_z(BATCH_SIZE, DIM_Z)}
        )

    _, loss_GGAN_G1 = sess.run(
        [G1_solver, G1_loss],
        feed_dict={z0: sample_z(BATCH_SIZE, DIM_Z)}
    )

    # Train 2nd-stage GoGAN
    if it >= N_STAGE1:
        for _ in range(5):
            batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)

            _, loss_GGAN_D2, _ = sess.run(
                [D2_solver, D2_loss, clip_D2],
                feed_dict={x0: batch_xs, z0: sample_z(BATCH_SIZE, DIM_Z)}
            )

        _, loss_GGAN_G2 = sess.run(
            [G2_solver, G2_loss],
            feed_dict={z0: sample_z(BATCH_SIZE, DIM_Z)}
        )
    else:
        loss_GGAN_D2 = 0
        loss_GGAN_G2 = 0

    plt.ion()
    if it % 100 == 0:
        print('{:10d}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}') \
                .format(it, loss_WGAN_D, loss_WGAN_G, loss_GGAN_D1, loss_GGAN_G1, loss_GGAN_D2, loss_GGAN_G2)

        if it % 1000 == 0:
            for i, output in enumerate(outputs):
                samples = sess.run(output, feed_dict={z0: sample_z(16, DIM_Z)})
                figs[i] = plot(samples, i)
                figs[i].canvas.draw()

                plt.savefig('out/' + fig_names[i].format(it / 1000), bbox_inches='tight')