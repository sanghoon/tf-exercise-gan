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
from models import *

DIM_Z = 32

W_CLIP = 0.01
BATCH_SIZE = 100

# Load dataset
data = input_data.read_data_sets('data/mnist/', one_hot=True)


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

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def_gen = simple_gen
def_dis = lambda x, name, **kwargs: simple_net(x, name, 1, **kwargs)


# Instantiate network
z0 = tf.placeholder(tf.float32, shape=[None, DIM_Z])
x0 = tf.placeholder(tf.float32, shape=[None, 784])
x1 = tf.reshape(x0, [-1,28,28,1])

global_step = tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(0.001, global_step, 500 * 10, 0.96, staircase=True)   # *0.96 per 10 epochs
lr = tf.constant(0.0001)
'''
Hexa's comments
- Unlike classification problems, using a fixed/small LR leads to better results in GAN trainings
  e.g.) 1e-6 for Cifar-10, 10e-4 for MNIST
- # of training steps is also important. You should investigate the output images with Tensorboard
'''
increment_step = tf.assign_add(global_step, 1)

# TODO: Refactoring
### WGAN
G = def_gen(z0, 'WGAN_G', bn=False)
D_real = def_dis(x1, 'WGAN_D', bn=False)
D_fake = def_dis(G, 'WGAN_D', bn=False, reuse=True)

# Loss functions
D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=lr))  \
            .minimize(D_loss, var_list=get_trainable_params('WGAN_D'))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=lr))  \
            .minimize(G_loss, var_list=get_trainable_params('WGAN_G'))

clip_D = [p.assign(tf.clip_by_value(p, -W_CLIP, W_CLIP))
            for p in get_trainable_params('WGAN_D')]


### GoGAN
epsilon = 0.5   # Margin
l1 = 1.0        # Disc. loss
l2 = 0.5        # Rank. loss

G1 = def_gen(z0, 'GoGAN_G1', bn=False)
D1_real = def_dis(x1, 'GoGAN_D1', bn=False)
D1_fake = def_dis(G1, 'GoGAN_D1', bn=False, reuse=True)

D1_loss = tf.reduce_mean(tf.nn.relu(D1_fake + epsilon - D1_real))
G1_loss = -tf.reduce_mean(D1_fake)

clip_D1 = [p.assign(tf.clip_by_value(p, -W_CLIP, W_CLIP))
            for p in get_trainable_params('GoGAN_D1')]

G2 = def_gen(z0, 'GoGAN_G2', bn=False)
D2_real = def_dis(x1, 'GoGAN_D2', bn=False)
D2_fake = def_dis(G2, 'GoGAN_D2', bn=False, reuse=True)

D2_loss = tf.reduce_mean(tf.nn.relu(D2_fake + epsilon - D2_real)) * l1 \
        + tf.reduce_mean(tf.nn.relu(D1_fake + 2 * epsilon - D2_real)) * l2
G2_loss = -tf.reduce_mean(D2_fake)

clip_D2 = [p.assign(tf.clip_by_value(p, -W_CLIP, W_CLIP))
            for p in get_trainable_params('GoGAN_D2')]

G1_solver = (tf.train.RMSPropOptimizer(learning_rate=lr)) \
            .minimize(G1_loss, var_list=get_trainable_params('GoGAN_G1'))
D1_solver = (tf.train.RMSPropOptimizer(learning_rate=lr)) \
            .minimize(D1_loss, var_list=get_trainable_params('GoGAN_D1'))
G2_solver = (tf.train.RMSPropOptimizer(learning_rate=lr)) \
            .minimize(G2_loss, var_list=get_trainable_params('GoGAN_G2'))
D2_solver = (tf.train.RMSPropOptimizer(learning_rate=lr)) \
            .minimize(D2_loss, var_list=get_trainable_params('GoGAN_D2'))


copy_G = ops_copy_vars(src_scope='GoGAN_G1', dst_scope='GoGAN_G2')
copy_D = ops_copy_vars(src_scope='GoGAN_D1', dst_scope='GoGAN_D2')


# Session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))     # XXX: Is this necessary?
#sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Initial setup for visualization
outputs = [G, G1, G2]
figs = [None] * len(outputs)
fig_names = ['fig_WGAN_gen_{:04d}.png', 'fig_GGAN_1st_{:04d}.png', 'fig_GGAN_2nd_{:04d}.png']
output_names = ['WGAN', 'GoGAN_1st', 'GoGAN_2nd']

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
        sess.run(copy_G)
        sess.run(copy_D)
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


    # Increment steps
    _, cur_lr = sess.run([increment_step, lr])

    plt.ion()
    if it % 100 == 0:
        print('{:10d}, {:1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}') \
                .format(it, cur_lr, loss_WGAN_D, loss_WGAN_G, loss_GGAN_D1, loss_GGAN_G1, loss_GGAN_D2, loss_GGAN_G2)

        rand_latent = sample_z(16, DIM_Z)
        if it % 1000 == 0:
            for i, output in enumerate(outputs):
                samples = sess.run(output, feed_dict={z0: rand_latent})
                figs[i] = plot(samples, i)
                figs[i].canvas.draw()

                plt.savefig('out/' + fig_names[i].format(it / 1000), bbox_inches='tight')
