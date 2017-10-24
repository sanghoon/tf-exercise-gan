#!/usr/bin/env python
# Tensorflow impl. of MAD-GAN (The 2nd alt.)

from tensorflow.examples.tutorials.mnist import input_data
from common import *
from datasets import data_celeba, data_mnist
from models.celeba_models import *
from models.mnist_models import *
from eval_funcs import *


def train_madgan(data, g_net, d_net, name='MADGAN',
                 dim_z=128, n_iters=1e5, lr=1e-4, batch_size=128,
                 sampler=sample_z, eval_funcs=[],
                 n_generators=4):

    ### 0. Common preparation
    hyperparams = {'NGEN': n_generators, 'LR': lr}
    base_dir, out_dir, log_dir = create_dirs(name, g_net.name, d_net.name, hyperparams)

    tf.reset_default_graph()

    global_step = tf.Variable(0, trainable=False)
    increment_step = tf.assign_add(global_step, 1)
    lr = tf.constant(lr)

    assert (batch_size % n_generators == 0)


    ### 1. Define network structure
    x_shape = data.train.images[0].shape
    z0 = tf.placeholder(tf.float32, shape=[None, dim_z])            # Latent var.
    x0 = tf.placeholder(tf.float32, shape=(None,) + x_shape)        # Generated images

    zs = tf.split(z0, num_or_size_splits=n_generators, axis=0)      # Across batch
    Gs = []
    for i in range(n_generators):
        # Common layers
        feat = g_net.former(zs[i], 'MADGAN_G', reuse=True if i > 0 else False)

        # Separated layers
        out = g_net.latter(feat, 'MADGAN_G{}'.format(i))
        Gs.append(out)

        # TODO: (experiments) How about sharing later layers only?

    G = tf.concat(Gs, 0)                    # As a single batch

    D1 = d_net(x0, 'MADGAN_D')
    D2 = d_net(G, 'MADGAN_D', reuse=True)
    D_batch = tf.concat([D1, D2], 0)        # Across batch

    D_fake = tf.nn.softmax(D2)[:, 0]       # If this is high, G(z) are predicted as real samples

    # Class labels
    # TODO: Make this stochastic
    n_repeat = batch_size // n_generators
    gt_list = [0] * batch_size + [i+1 for i in range(n_generators) for n in range(n_repeat)]  # 0, ... , 0, 1, 1, 2, 2, ...
    #y0 = tf.Variable(tf.one_hot(gt_list, n_generators + 1))     # one-hot encoding of generator labels (0: real)

    # Loss functions
    #D_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_batch, labels=y0))
    D_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_batch, labels=gt_list))
    G_loss = tf.reduce_mean(-tf.log(D_fake))

    D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)) \
        .minimize(D_loss, var_list=get_trainable_params('MADGAN_D'))
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)) \
        .minimize(G_loss, var_list=get_trainable_params('MADGAN_G'))


    #### 2. Operations for log/state back-up
    tf.summary.scalar('MADGAN_D_loss', D_loss)
    tf.summary.scalar('MADGAN_G_loss', G_loss)

    if check_dataset_type(x_shape) != 'synthetic':
        tf.summary.image('MADGAN', G, max_outputs=4)        # for images only

    summaries = tf.summary.merge_all()

    saver = tf.train.Saver(get_trainable_params('MADGAN_D') + get_trainable_params('MADGAN_G'))

    # Initial setup for visualization
    outputs = [G]
    figs = [None] * len(outputs)
    fig_names = ['fig_gen_{:04d}_MADGAN.png']

    plt.ion()


    ### 3. Run a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print('{:>10}, {:>7}, {:>7}, {:>7}') \
        .format('Iters', 'cur_LR', 'MADGAN_D', 'MADGAN_G')


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

        _, cur_lr = sess.run([increment_step, lr])

        if it % PRNT_INTERVAL == 0:
            print('{:10d}, {:1.4f}, {: 1.4f}, {: 1.4f}') \
                    .format(it, cur_lr, loss_D, loss_G)

            # Tensorboard
            cur_summary = sess.run(summaries, feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)})
            writer.add_summary(cur_summary, it)

        if it % SHOW_FIG_INTERVAL == 0:
            # FIXME
            img_generator = lambda n: sess.run(output, feed_dict={z0: sampler(n, dim_z)})

            for i, output in enumerate(outputs):
                figs[i] = data.plot(img_generator, fig_id=i, batch_size = batch_size)
                figs[i].canvas.draw()
	        if it % EVAL_INTERVAL == 0:
                    plt.savefig(out_dir + fig_names[i].format(it / 1000), bbox_inches='tight')

            # Run evaluation functions
            if it % EVAL_INTERVAL == 0:
                for func in eval_funcs:
                    func(it, img_generator)

        if it % SAVE_INTERVAL == 0:
            saver.save(sess, out_dir + 'madgan', it)

    sess.close()


if __name__ == '__main__':
    args = parse_args(additional_args=[
        ('--n_gen', {'type': int, 'default': 4}),
    ])
    print args

    if args.gpu:
        set_gpu(args.gpu)

    if args.datasets == 'mnist':
        out_name = 'MADGAN_mnist'
        out_name = out_name if len(args.tag) == 0 else '{}_{}'.format(out_name, args.tag)

        dim_z = 64
        n_generators = args.n_gen

        data = data_mnist.MnistWrapper('datasets/mnist/')
        g_net = SimpleGEN(dim_z, last_act=tf.sigmoid)
        d_net = SimpleCNN(n_generators + 1)

        train_madgan(data, g_net, d_net, name=out_name,
                     dim_z=dim_z, n_generators=n_generators, batch_size=args.batchsize, lr=args.lr,
                     eval_funcs=[lambda it, gen: eval_images_naive(it, gen, data)])

    elif args.datasets == 'celeba':
        out_name = 'MADGAN_celeba'
        out_name = out_name if len(args.tag) == 0 else '{}_{}'.format(out_name, args.tag)

        dim_z = 128
        n_generators = args.n_gen

        data = data_celeba.CelebA('datasets/img_align_celeba')
        g_net = DCGAN_G(dim_z, last_act=tf.sigmoid)
        d_net = DCGAN_D(n_generators + 1)

        train_madgan(data, g_net, d_net, name=out_name,
                     dim_z=dim_z, n_generators=n_generators, batch_size=args.batchsize, lr=args.lr,
                     eval_funcs=[lambda it, gen: eval_images_naive(it, gen, data)])
