#!/usr/bin/env/python
# Tensorflow impl. of MAD-GAN

from tensorflow.examples.tutorials.mnist import input_data
from common import *
from models.mnist_models import *
from models.celeba_models import *
import data_celeba


def train_madgan(data, g_net, d_net, tag='',
                 dim_z=128, n_iters=2e5, lr=1e-4, batch_size=128, n_generators=8, eval_func=None):
    name = 'MADGAN'
    n_iters = int(n_iters)

    # TODO: Folder generator
    BASE_FOLDER = 'out/{}_{}_{}_{}/MA{}_LR{}/'.format(name, tag, g_net.name, d_net.name, n_generators, lr)
    OUT_FOLDER = os.path.join(BASE_FOLDER, 'out/')
    LOG_FOLDER = os.path.join(BASE_FOLDER, 'log/')
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    assert (batch_size % n_generators == 0)

    # Define network
    h, w, c = data.train.images[0].shape
    z0 = tf.placeholder(tf.float32, shape=[None, dim_z])
    x0 = tf.placeholder(tf.float32, shape=[None, h, w, c])

    # Step and LR
    global_step = tf.Variable(0, trainable=False)
    increment_step = tf.assign_add(global_step, 1)
    lr = tf.constant(lr)

    # Network structure
    zs = tf.split(z0, num_or_size_splits=n_generators, axis=0)      # Across batch
    Gs = []
    for i in range(n_generators):
        # Common layers
        feat = g_net.former(zs[i], 'MADGAN_G', reuse=True if i > 0 else False)

        # Separated layers
        out = g_net.latter(feat, 'MADGAN_G{}'.format(i))
        Gs.append(out)

        # TODO: How about sharing later layers only?
    G = tf.concat(Gs, 0)                    # As a single batch

    D1 = d_net(x0, 'MADGAN_D')
    D2 = d_net(G, 'MADGAN_D', reuse=True)
    D_batch = tf.concat([D1, D2], 0)        # Across batch

    D_fake = tf.nn.softmax(D2)[:, 0]       # If this is high, G(z) are predicted as real samples

    # Class labels
    # TODO: Make this stochastic
    n_repeat = batch_size // n_generators
    gt_list = [0] * batch_size + [n for i in range(n_repeat) for n in range(n_generators)]  # 0, ... , 0, 1, 1, 2, 2, ...
    y0 = tf.Variable(tf.one_hot(gt_list, n_generators + 1))     # one-hot encoding of generator labels (0: real)

    # Loss functions
    D_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_batch, labels=y0))
    G_loss = tf.reduce_mean(-tf.log(D_fake))

    D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)) \
        .minimize(D_loss, var_list=get_trainable_params('MADGAN_D'))
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)) \
        .minimize(G_loss, var_list=get_trainable_params('MADGAN_G'))

    # Tensorboard
    tf.summary.scalar('MADGAN_D_loss', D_loss)
    tf.summary.scalar('MADGAN_G_loss', G_loss)
    tf.summary.image('MADGAN', G, max_outputs=4)
    summaries = tf.summary.merge_all()


    # Session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(LOG_FOLDER, sess.graph)

    # Initial setup for visualization
    outputs = [G]
    figs = [None] * len(outputs)
    fig_names = ['fig_gen_{:04d}_MADGAN.png']

    saver = tf.train.Saver(get_trainable_params('MADGAN_D') + get_trainable_params('MADGAN_G'))

    print('{:>10}, {:>7}, {:>7}, {:>7}') \
        .format('Iters', 'cur_LR', 'MADGAN_D', 'MADGAN_G')


    for it in range(n_iters):
        # Train MADGAN
        batch_xs, _ = data.train.next_batch(batch_size)

        _, loss_D = sess.run(
            [D_solver, D_loss],
            feed_dict={x0: batch_xs, z0: sample_z(batch_size, dim_z)}
        )

        _, loss_G = sess.run(
            [G_solver, G_loss],
            feed_dict={z0: sample_z(batch_size, dim_z)}
        )

        ## Increment steps
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
            saver.save(sess, OUT_FOLDER + 'madgan', it)

            # TODO: Check precision
            if eval_func:
                rand_latent = sample_z(1024, dim_z)
                samples = sess.run
                eval_func(samples)



if __name__ == '__main__':
    args = parse_args(additional_args=[
        ('--n_gen', {'type': int, 'default': 8})
    ,])
    print args

    if args.gpu:
        set_gpu(args.gpu)

    if args.data == 'mnist':
        dim_z = 64
        n_generators = args.n_gen

        data = input_data.read_data_sets('data/mnist/', one_hot=True, reshape=False)
        g_net = SimpleGEN(dim_z, last_act=tf.sigmoid)
        d_net = SimpleCNN(n_generators + 1)

        train_madgan(data, g_net, d_net, dim_z=dim_z, n_generators=n_generators, batch_size=args.batchsize, lr=args.lr)

    elif args.data == 'celeba':
        dim_z = 128
        n_generators = args.n_gen

        data = data_celeba.CelebA('data/img_align_celeba')
        g_net = DCGAN_G(dim_z, last_act=tf.tanh)        # Used identity instead of tanh
        d_net = DCGAN_D(n_generators + 1)

        train_madgan(data, g_net, d_net, dim_z=dim_z, n_generators=n_generators, batch_size=args.batchsize, lr=args.lr)