#!/usr/bin/env python
from tensorflow.examples.tutorials.mnist import input_data
import data_celeba
from common import *
from models.models import *
from models.mnist_models import *
from models.celeba_models import *


def train_dcgan(data, g_net, d_net, tag='',
                dim_z=128, n_iters=2e5, lr=1e-4, batch_size=128, eval_func=None):
    name = 'DCGAN'
    n_iters = int(n_iters)

    # TODO: Folder generator
    BASE_FOLDER = 'out/{}_{}_{}_{}/LR{}/'.format(name, tag, g_net.name, d_net.name, lr)
    OUT_FOLDER = os.path.join(BASE_FOLDER, 'out/')
    LOG_FOLDER = os.path.join(BASE_FOLDER, 'log/')
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)


    # Define network
    h, w, c = data.train.images[0].shape
    z0 = tf.placeholder(tf.float32, shape=[None, dim_z])
    x0 = tf.placeholder(tf.float32, shape=[None, h, w, c])

    # Step and LR
    global_step = tf.Variable(0, trainable=False)
    increment_step = tf.assign_add(global_step, 1)
    lr = tf.constant(lr)


    ### DCGAN
    G = g_net(z0, 'DCGAN_G')
    D_real = d_net(x0, 'DCGAN_D')
    D_fake = d_net(G, 'DCGAN_D', reuse=True)

    # Loss functions
    D_loss = tf.reduce_mean(-tf.log(D_real)-tf.log(1-D_fake))
    G_loss = tf.reduce_mean(-tf.log(D_fake))

    D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
                .minimize(D_loss, var_list=get_trainable_params('DCGAN_D'))
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
                .minimize(G_loss, var_list=get_trainable_params('DCGAN_G'))

    # Tensorboard
    tf.summary.scalar('DCGAN_D(x)', tf.reduce_mean(D_real))
    tf.summary.scalar('DCGAN_D(G)', tf.reduce_mean(D_fake))
    tf.summary.scalar('DCGAN_D_loss', tf.reduce_mean(D_loss))
    tf.summary.scalar('DCGAN_G_loss', tf.reduce_mean(G_loss))
    tf.summary.image('DCGAN', G, max_outputs=3)
    summaries = tf.summary.merge_all()


    # Session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(LOG_FOLDER, sess.graph)

    # Initial setup for visualization
    outputs = [G]
    figs = [None] * len(outputs)
    fig_names = ['fig_gen_{:04d}_DCGAN.png']

    saver = tf.train.Saver(get_trainable_params('DCGAN_D') + get_trainable_params('DCGAN_G'))


    print('{:>10}, {:>7}, {:>7}, {:>7}') \
        .format('Iters', 'cur_LR', 'DCGAN_D', 'DCGAN_G')


    for it in range(n_iters):
        # Train DCGAN
        batch_xs, batch_ys = data.train.next_batch(batch_size)

        _, loss_D = sess.run(
                [D_solver, D_loss],
                feed_dict={x0: batch_xs, z0: sample_z(batch_size, dim_z)}
            )

        _, loss_G = sess.run(
            [G_solver, G_loss],
            feed_dict={z0: sample_z(batch_size, dim_z)}
        )

        # Increment steps
        _, cur_lr = sess.run([increment_step, lr])

        plt.ion()
        if it % 100 == 0:
            print('{:10d}, {:1.4f}, {: 1.4f}, {: 1.4f}') \
                    .format(it, cur_lr, loss_D, loss_G)

            rand_latent = sample_z(16, dim_z)

            if it % 1000 == 0:
                for i, output in enumerate(outputs):
                    samples = sess.run(output, feed_dict={z0: rand_latent})
                    figs[i] = plot(samples, i, shape=(h, w, c))
                    figs[i].canvas.draw()

                    plt.savefig(OUT_FOLDER + fig_names[i].format(it / 1000), bbox_inches='tight')

            # Tensorboard
            cur_summary = sess.run(summaries, feed_dict={x0: batch_xs, z0: sample_z(batch_size, dim_z)})
            writer.add_summary(cur_summary, it)

        if it % 10000 == 0:
            saver.save(sess, OUT_FOLDER + 'dcgan', it)


if __name__ == '__main__':
    args = parse_args()
    print args

    if args.gpu:
        set_gpu(args.gpu)

    if args.data == 'mnist':
        dim_z = 64

        data = input_data.read_data_sets('data/mnist/', one_hot=True, reshape=False)
        g_net = SimpleGEN(dim_z, last_act=tf.sigmoid)
        d_net = SimpleCNN(1, last_act=tf.sigmoid)

        train_dcgan(data, g_net, d_net, tag='mnist', dim_z=dim_z,  batch_size=args.batchsize, lr=args.lr)

    elif args.data == 'celeba':
        dim_z = 128

        data = data_celeba.CelebA('data/img_align_celeba')
        g_net = DCGAN_G(dim_z, last_act=tf.tanh)        # Used identity instead of tanh
        d_net = DCGAN_D(1, last_act=tf.sigmoid)

        train_dcgan(data, g_net, d_net, tag='celeba', dim_z=dim_z, batch_size=args.batchsize, lr=args.lr)