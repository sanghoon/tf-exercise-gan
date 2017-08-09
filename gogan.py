#!/usr/bin/env python
# Tensorflow impl. of GoGAN

from common import *
from datasets import data_celeba, data_mnist
from models.celeba_models import *
from models.mnist_models import *
from eval_funcs import *

def train_gogan(data, g_net, d_net, name='GoGAN',
                 dim_z=128, n_iters=1e5, lr=1e-4, batch_size=128,
                 sampler=sample_z, eval_funcs=[],
                 w_clip=0.1, epsilon=1.0, l_disc=1.0, l_rank=0.5):

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

    # 1st stage
    G1 = g_net(z0, 'GoGAN_G1')
    D1_real = d_net(x0, 'GoGAN_D1')
    D1_fake = d_net(G1, 'GoGAN_D1', reuse=True)

    D1_loss = tf.reduce_mean(tf.nn.relu(D1_fake + epsilon - D1_real))
    G1_loss = -tf.reduce_mean(D1_fake)

    clip_D1 = [p.assign(tf.clip_by_value(p, -w_clip, w_clip))
                for p in get_trainable_params('GoGAN_D1')]

    G1_solver = (tf.train.RMSPropOptimizer(learning_rate=lr)) \
                .minimize(G1_loss, var_list=get_trainable_params('GoGAN_G1'))
    D1_solver = (tf.train.RMSPropOptimizer(learning_rate=lr)) \
                .minimize(D1_loss, var_list=get_trainable_params('GoGAN_D1'))

    # 2nd stage
    G2 = g_net(z0, 'GoGAN_G2')
    D2_real = d_net(x0, 'GoGAN_D2')
    D2_fake = d_net(G2, 'GoGAN_D2', reuse=True)

    D2_loss = tf.reduce_mean(tf.nn.relu(D2_fake + epsilon - D2_real)) * l_disc \
            + tf.reduce_mean(tf.nn.relu(D1_fake + 2 * epsilon - D2_real)) * l_rank
    G2_loss = -tf.reduce_mean(D2_fake)

    clip_D2 = [p.assign(tf.clip_by_value(p, -w_clip, w_clip))
                for p in get_trainable_params('GoGAN_D2')]

    G2_solver = (tf.train.RMSPropOptimizer(learning_rate=lr)) \
                .minimize(G2_loss, var_list=get_trainable_params('GoGAN_G2'))
    D2_solver = (tf.train.RMSPropOptimizer(learning_rate=lr)) \
                .minimize(D2_loss, var_list=get_trainable_params('GoGAN_D2'))

    # Copy operation from level1 to level2
    copy_G = ops_copy_vars(src_scope='GoGAN_G1', dst_scope='GoGAN_G2')
    copy_D = ops_copy_vars(src_scope='GoGAN_D1', dst_scope='GoGAN_D2')


    #### 2. Operations for log/state back-up
    tf.summary.scalar('GGAN_D1(x)', tf.reduce_mean(D1_real))
    tf.summary.scalar('GGAN_D1(G)', tf.reduce_mean(D1_fake))
    tf.summary.scalar('GGAN_D2(x)', tf.reduce_mean(D2_real))
    tf.summary.scalar('GGAN_D2(G)', tf.reduce_mean(D2_fake))

    # Output images
    if check_dataset_type(x_shape) != 'synthetic':
        tf.summary.image('GoGAN_1st', G1, max_outputs=3)
        tf.summary.image('GoGAN_2nd', G2, max_outputs=3)

    summaries = tf.summary.merge_all()

    saver = tf.train.Saver(get_trainable_params('WGAN') + get_trainable_params('GoGAN'))

    # Initial setup for visualization
    outputs = [G1, G2]
    figs = [None] * len(outputs)
    fig_names = ['fig_GGAN_1st_{:04d}.png', 'fig_GGAN_2nd_{:04d}.png']

    plt.ion()

    ### 3. Run a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print('{:>10}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}') \
        .format('Iters', 'cur_LR', 'GGAN_D1', 'GGAN_G1', 'GGAN_D2', 'GGAN_G2')

    n_iters_before_stage2 = n_iters // 2
    for it in range(int(n_iters)):
        if it == n_iters_before_stage2:
            # Copy vars. and start phase-2
            sess.run(copy_G)
            sess.run(copy_D)
            pass

        # Train 1st-stage GoGAN
        for _ in range(5):
            batch_xs, batch_ys = data.train.next_batch(batch_size)

            _, loss_GGAN_D1, _ = sess.run(
                [D1_solver, D1_loss, clip_D1],
                feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)}
            )

        _, loss_GGAN_G1 = sess.run(
            [G1_solver, G1_loss],
            feed_dict={z0: sampler(batch_size, dim_z)}
        )

        # Train 2nd-stage GoGAN
        if it >= n_iters_before_stage2:
            for _ in range(5):
                batch_xs, batch_ys = data.train.next_batch(batch_size)

                _, loss_GGAN_D2, _ = sess.run(
                    [D2_solver, D2_loss, clip_D2],
                    feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)}
                )

            _, loss_GGAN_G2 = sess.run(
                [G2_solver, G2_loss],
                feed_dict={z0: sampler(batch_size, dim_z)}
            )
        else:
            loss_GGAN_D2 = 0
            loss_GGAN_G2 = 0


        _, cur_lr = sess.run([increment_step, lr])

        if it % PRNT_INTERVAL == 0:
            print('{:10d}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}') \
                    .format(it, cur_lr, loss_GGAN_D1, loss_GGAN_G1, loss_GGAN_D2, loss_GGAN_G2)

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
        out_name = 'GoGAN_mnist'
        out_name = out_name if len(args.tag) == 0 else '{}_{}'.format(out_name, args.tag)

        dim_z = 64

        data = data_mnist.MnistWrapper('datasets/mnist/')

        g_net = SimpleGEN(dim_z, last_act=tf.sigmoid)
        d_net = SimpleCNN(n_out=1, last_act=tf.identity)

        train_gogan(data, g_net, d_net, name=out_name, dim_z=dim_z,  batch_size=args.batchsize, lr=args.lr,
                    w_clip=args.w_clip,
                    eval_funcs=[lambda it, gen: eval_images_naive(it, gen, data)])


    elif args.datasets == 'celeba':
        out_name = 'GoGAN_celeba'
        out_name = out_name if len(args.tag) == 0 else '{}_{}'.format(out_name, args.tag)

        dim_z = 128
        dim_h = 64

        data = data_celeba.CelebA('datasets/img_align_celeba')

        g_net = DCGAN_G(dim_z, last_act=tf.sigmoid)
        d_net = DCGAN_D(n_out=1, last_act=tf.identity)

        train_gogan(data, g_net, d_net, name=out_name, dim_z=dim_z, batch_size=args.batchsize, lr=args.lr,
                    w_clip=args.w_clip,
                    eval_funcs=[lambda it, gen: eval_images_naive(it, gen, data)])