#!/usr/bin/env python
# Tensorflow impl. of WGAN

from common import *
from datasets import data_celeba, data_mnist
from models.celeba_models import *
from models.mnist_models import *
from eval_funcs import *

def train_wgan(data, g_net, d_net, name='WGAN',
                 dim_z=128, n_iters=1e5, lr=1e-4, batch_size=128,
                 sampler=sample_z, eval_funcs=[],
                 w_clip=0.1):

    ### 0. Common preparation
    hyperparams = {'LR': lr, 'WClip': w_clip}
    base_dir, out_dir, log_dir = create_dirs(name, g_net.name, d_net.name, hyperparams)

    tf.reset_default_graph()

    global_step = tf.Variable(0, trainable=False)
    increment_step = tf.assign_add(global_step, 1)
    lr = tf.constant(lr)

    ### 1. Define network structure
    x_shape = data.train.images[0].shape
    z0 = tf.placeholder(tf.float32, shape=[None, dim_z])        # Latent var.
    x0 = tf.placeholder(tf.float32, shape=(None,) + x_shape)    # Generated images

    ### define WGAN
    G = g_net(z0, 'WGAN_G')
    D_real = d_net(x0, 'WGAN_D')
    D_fake = d_net(G, 'WGAN_D', reuse=True)

    # Loss functions
    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
    G_loss = -tf.reduce_mean(D_fake)

    D_solver = (tf.train.RMSPropOptimizer(learning_rate=lr))  \
                .minimize(D_loss, var_list=get_trainable_params('WGAN_D'))
    G_solver = (tf.train.RMSPropOptimizer(learning_rate=lr))  \
                .minimize(G_loss, var_list=get_trainable_params('WGAN_G'))

    clip_D = [p.assign(tf.clip_by_value(p, -w_clip, w_clip))
                for p in get_trainable_params('WGAN_D')]


    #### 2. Operations for log/state back-up
    tf.summary.scalar('WGAN_D(x)', tf.reduce_mean(D_real))
    tf.summary.scalar('WGAN_D(G)', tf.reduce_mean(D_fake))
    tf.summary.scalar('WGAN_D_loss', tf.reduce_mean(D_loss))
    tf.summary.scalar('WGAN_G_loss', tf.reduce_mean(G_loss))

    # Output images
    if check_dataset_type(x_shape) != 'synthetic':
        tf.summary.image('WGAN', G, max_outputs=3)

    summaries = tf.summary.merge_all()

    saver = tf.train.Saver(get_trainable_params('WGAN') + get_trainable_params('GoGAN'))

    # Initial setup for visualization
    outputs = [G]
    figs = [None] * len(outputs)
    fig_names = ['fig_WGAN_gen_{:04d}.png']

    plt.ion()

    ### 3. Run a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print('{:>10}, {:>7}, {:>7}, {:>7}') \
        .format('Iters', 'cur_LR', 'WGAN_D', 'WGAN_G')

    for it in range(int(n_iters)):
        for _ in range(5):
            batch_xs, batch_ys = data.train.next_batch(batch_size)

            _, loss_WGAN_D, _ = sess.run(
                [D_solver, D_loss, clip_D],
                feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)}
            )

        _, loss_WGAN_G = sess.run(
            [G_solver, G_loss],
            feed_dict={z0: sampler(batch_size, dim_z)}
        )

        _, cur_lr = sess.run([increment_step, lr])

        if it % PRNT_INTERVAL == 0:
            print('{:10d}, {:1.4f}, {: 1.4f}, {: 1.4f}') \
                    .format(it, cur_lr, loss_WGAN_D, loss_WGAN_G)

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
            saver.save(sess, out_dir + 'gogan', it)

    sess.close()


if __name__ == '__main__':
    args = parse_args(additional_args=[
        ('--w_clip', {'type': float, 'default': 0.1}),
    ])
    print args

    if args.gpu:
        set_gpu(args.gpu)

    if args.datasets == 'mnist':
        out_name = 'WGAN_mnist'
        out_name = out_name if len(args.tag) == 0 else '{}_{}'.format(out_name, args.tag)

        dim_z = 64

        data = data_mnist.MnistWrapper('datasets/mnist/')

        g_net = SimpleGEN(dim_z, last_act=tf.sigmoid)
        d_net = SimpleCNN(n_out=1, last_act=tf.identity)

        train_wgan(data, g_net, d_net, name=out_name, dim_z=dim_z,  batch_size=args.batchsize, lr=args.lr,
                   w_clip=args.w_clip,
                   eval_funcs=[lambda it, gen: eval_images_naive(it, gen, data)])


    elif args.datasets == 'celeba':
        out_name = 'WGAN_celeba'
        out_name = out_name if len(args.tag) == 0 else '{}_{}'.format(out_name, args.tag)

        dim_z = 128

        data = data_celeba.CelebA('datasets/img_align_celeba')

        g_net = DCGAN_G(dim_z, last_act=tf.tanh)
        d_net = DCGAN_D(n_out=1, last_act=tf.identity)

        train_wgan(data, g_net, d_net, name=out_name, dim_z=dim_z, batch_size=args.batchsize, lr=args.lr,
                   eval_funcs=[lambda it, gen: eval_images_naive(it, gen, data)])