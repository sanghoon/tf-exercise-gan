#!/usr/bin/env python
import os
from tensorflow.examples.tutorials.mnist import input_data
from models import *
from utils import *
from common import *


args = parse_args(models.keys())

print args

if len(args.tag) == 0:
    args.tag = 'began'

if args.net is None:
    args.net = 'simple_cnn'

BASE_FOLDER = 'out_{}/{}_BN{}_LR{}_K{}/'.format(args.tag, args.net, int(args.bn), args.lr, args.kernel)
OUT_FOLDER = os.path.join(BASE_FOLDER, 'out/')
LOG_FOLDER = os.path.join(BASE_FOLDER, 'log/')

assert('cnn' in args.net)

def began_disc(x, name, dim_h=32, k=args.kernel, **kwargs):
    h0 = models[args.net][1](x, name + '/enc', n_out=dim_h, last_act=tf.identity, k=k, **kwargs)
    x_ = models[args.net][0](h0, name + '/dec', n_in=dim_h, last_act=tf.sigmoid, k=k, **kwargs)

    out = tf.reduce_mean(tf.reduce_sum((x - x_) ** 2, 1))

    return out

def_gen = lambda x, name, **kwargs: models[args.net][0](x, name, k=args.kernel, **kwargs)
def_dis = lambda x, name, **kwargs: began_disc(x, name, **kwargs)



LR = args.lr

z0 = tf.placeholder(tf.float32, shape=[None, DIM_Z])
x0 = tf.placeholder(tf.float32, shape=[None, 784])
x1 = tf.reshape(x0, [-1,28,28,1])
k = tf.Variable(0.0, trainable=False)
l_k = 0.001
g_k = 0.5

global_step = tf.Variable(0, trainable=False)
increment_step = tf.assign_add(global_step, 1)

lr = tf.constant(LR)



### BEGAN
G = def_gen(z0, 'BEGAN_G', bn=args.bn)
D_real = def_dis(x1, 'BEGAN_D', bn=args.bn)
D_fake = def_dis(G, 'BEGAN_D', bn=args.bn, reuse=True)

# Loss functions
D_loss = tf.reduce_mean(D_real - k * D_fake)
G_loss = tf.reduce_mean(D_fake)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
            .minimize(D_loss, var_list=get_trainable_params('BEGAN_D'))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
            .minimize(G_loss, var_list=get_trainable_params('BEGAN_G'))

tf.summary.scalar('BEGAN_D(x)', tf.reduce_mean(D_real))
tf.summary.scalar('BEGAN_D(G)', tf.reduce_mean(D_fake))
tf.summary.scalar('k', k)


# Convergence metric
M = D_real + tf.abs(g_k * D_real - D_fake)

tf.summary.scalar('M', M)


# update k
update_k = k.assign(k + l_k * (g_k * D_real - D_fake))



# Output images
tf.summary.image('BEGAN', G, max_outputs=3)

# Tensorboard
summaries = tf.summary.merge_all()


# Session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(LOG_FOLDER, sess.graph)

# Initial setup for visualization
outputs = [G]
figs = [None] * len(outputs)
fig_names = ['fig_BEGAN_gen_{:04d}.png']
output_names = ['BEGAN']

if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)



saver = tf.train.Saver(get_trainable_params('BEGAN_D') + get_trainable_params('BEGAN_G'))


# Load dataset
data = input_data.read_data_sets('data/mnist/', one_hot=True)

print('{:>10}, {:>7}, {:>7}, {:>7}') \
    .format('Iters', 'cur_LR', 'BEGAN_D', 'BEGAN_G')

# 500 iterations = 1 epoch
for it in range(N_ITERS):
    # Train DCGAN
    batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)

    _, loss_D = sess.run(
            [D_solver, D_loss],
            feed_dict={x0: batch_xs, z0: sample_z(BATCH_SIZE, DIM_Z)}
        )

    _, loss_G = sess.run(
        [G_solver, G_loss],
        feed_dict={z0: sample_z(BATCH_SIZE, DIM_Z)}
    )

    cur_k = sess.run(update_k, feed_dict={x0: batch_xs, z0: sample_z(BATCH_SIZE, DIM_Z)})

    # Increment steps
    _, cur_lr = sess.run([increment_step, lr])

    plt.ion()
    if it % 100 == 0:
        print('{:10d}, {:1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}') \
                .format(it, cur_lr, loss_D, loss_G, cur_k)

        rand_latent = sample_z(16, DIM_Z)

        # TODO: convergence_measure

        if it % 1000 == 0:
            for i, output in enumerate(outputs):
                samples = sess.run(output, feed_dict={z0: rand_latent})
                figs[i] = plot(samples, i)
                figs[i].canvas.draw()

                plt.savefig(OUT_FOLDER + fig_names[i].format(it / 1000), bbox_inches='tight')

        # Tensorboard
        cur_summary = sess.run(summaries, feed_dict={x0: batch_xs, z0: rand_latent})
        writer.add_summary(cur_summary, it)

        if it % 10000 == 0:
            saver.save(sess, OUT_FOLDER + 'began', it)

