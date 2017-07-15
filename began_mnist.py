#!/usr/bin/env python
# Tensorflow impl. of BEGAN


from common import *
from datasets import data_celeba, data_mnist
from models.celeba_models import *
from models.mnist_models import *


def train_began(data, g_net, d_enc, d_dec, name='BEGAN',
                 dim_z=128, n_iters=1e5, lr=1e-4, batch_size=128,
                 l_k = 0.001, g_k = 0.5,
                 sampler=sample_z, eval_funcs=[]):

    ### 0. Common preparation
    hyperparams = {'LR': lr}
    base_dir, out_dir, log_dir = create_dirs(name, g_net.name, d_enc.name, hyperparams)

    global_step = tf.Variable(0, trainable=False)
    increment_step = tf.assign_add(global_step, 1)
    lr = tf.constant(lr)

    ### 1. Define network structure
    x_shape = data.train.images[0].shape
    z0 = tf.placeholder(tf.float32, shape=[None, dim_z])        # Latent var.
    x0 = tf.placeholder(tf.float32, shape=(None,) + x_shape)    # Generated images

    # BEGAN-specific vars.
    k = tf.Variable(0.0, trainable=False)

    def began_disc(x, name, **kwargs):
        h0 = d_enc(x, name + '_ENC', **kwargs)
        x_ = d_dec(h0, name + '_DEC', **kwargs)

        out = tf.reduce_mean(tf.reduce_sum((x - x_) ** 2, 1))

        return out

    G = g_net(z0, 'BEGAN_G')
    D_real = began_disc(x0, 'BEGAN_D')
    D_fake = began_disc(G, 'BEGAN_D', reuse=True)

    # Loss functions
    D_loss = tf.reduce_mean(D_real - k * D_fake)
    G_loss = tf.reduce_mean(D_fake)

    D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
                .minimize(D_loss, var_list=get_trainable_params('BEGAN_D'))
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
                .minimize(G_loss, var_list=get_trainable_params('BEGAN_G'))

    # Convergence metric
    M = D_real + tf.abs(g_k * D_real - D_fake)

    # update k
    update_k = k.assign(k + l_k * (g_k * D_real - D_fake))


    #### 2. Operations for log/state back-up
    tf.summary.scalar('BEGAN_D(x)', tf.reduce_mean(D_real))
    tf.summary.scalar('BEGAN_D(G)', tf.reduce_mean(D_fake))
    tf.summary.scalar('BEGAN_D_loss', tf.reduce_mean(D_loss))
    tf.summary.scalar('BEGAN_G_loss', tf.reduce_mean(G_loss))
    tf.summary.scalar('k', k)
    tf.summary.scalar('M', M)

    if check_dataset_type(x_shape) != 'synthetic':
        tf.summary.image('BEGAN', G, max_outputs=4)

    summaries = tf.summary.merge_all()

    saver = tf.train.Saver(get_trainable_params('BEGAN_D') + get_trainable_params('BEGAN_G'))

    # Initial setup for visualization
    outputs = [G]
    figs = [None] * len(outputs)
    fig_names = ['fig_BEGAN_gen_{:04d}.png']

    plt.ion()

    ### 3. Run a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=False, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print('{:>10}, {:>7}, {:>7}, {:>7}') \
        .format('Iters', 'cur_LR', 'BEGAN_D', 'BEGAN_G')


    for it in range(int(n_iters)):
        batch_xs, batch_ys = data.train.next_batch(batch_size)

        _, loss_D = sess.run(
            [D_solver, D_loss],
            feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)}
        )

        _, loss_G = sess.run(
            [G_solver, G_loss],
            feed_dict={z0: sampler(batch_size, dim_z)}
        )

        cur_k = sess.run(update_k, feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)})

        _, cur_lr = sess.run([increment_step, lr])

        if it % PRNT_INTERVAL == 0:
            print('{:10d}, {:1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}') \
                    .format(it, cur_lr, loss_D, loss_G, cur_k)

            # Tensorboard
            cur_summary = sess.run(summaries, feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)})
            writer.add_summary(cur_summary, it)

        if it % EVAL_INTERVAL == 0:
            img_generator = lambda n: sess.run(output, feed_dict={z0: sampler(n, dim_z)})

            for i, output in enumerate(outputs):
                figs[i] = data.plot(img_generator, fig_id=i)
                figs[i].canvas.draw()

                plt.savefig(out_dir + fig_names[i].format(it / 1000), bbox_inches='tight')

            # Run evaluation functions
            for func in eval_funcs:
                func(it, img_generator)

        if it % SAVE_INTERVAL == 0:
            saver.save(sess, out_dir + 'began', it)


if __name__ == '__main__':
    args = parse_args(additional_args=[])
    print args

    if args.gpu:
        set_gpu(args.gpu)

    if args.datasets == 'mnist':
        dim_z = 64
        dim_h = 16

        data = data_mnist.MnistWrapper('datasets/mnist/')

        # BEGAN doesn't seem to work well with BN
        g_net = SimpleGEN(dim_z, last_act=tf.sigmoid, bn=False)
        d_enc = SimpleCNN(n_out=dim_h, last_act=tf.identity, bn=False)
        d_dec = SimpleGEN(n_in=dim_h, last_act=tf.sigmoid, bn=False)

        train_began(data, g_net, d_enc, d_dec, name='BEGAN_mnist', dim_z=dim_z,  batch_size=args.batchsize, lr=args.lr)


    elif args.datasets == 'celeba':
        dim_z = 128
        dim_h = 64

        data = data_celeba.CelebA('datasets/img_align_celeba')

        # BEGAN doesn't seem to work well with BN
        g_net = DCGAN_G(dim_z, last_act=tf.tanh, bn=False)
        d_enc = DCGAN_D(n_out=dim_h, last_act=tf.identity, bn=False)
        d_dec = DCGAN_G(n_in=dim_h, last_act=tf.tanh, bn=False)

        train_began(data, g_net, d_enc, d_dec, name='BEGAN_celeba', dim_z=dim_z, batch_size=args.batchsize, lr=args.lr)