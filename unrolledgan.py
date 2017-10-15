#!/usr/bin/env/python
# Tensorflow impl. of DCGAN

from tensorflow.examples.tutorials.mnist import input_data
from common import *
from datasets import data_celeba, data_mnist
from models.celeba_models import *
from models.mnist_models import *
from eval_funcs import *
from keras.optimizers import Adam


_graph_replace = tf.contrib.graph_editor.graph_replace

def remove_original_op_attributes(graph):
    """Remove _original_op attribute from all operations in a graph."""
    for op in graph.get_operations():
        op._original_op = None

def graph_replace(*args, **kwargs):
    """Monkey patch graph_replace so that it works with TF 1.0"""
    remove_original_op_attributes(tf.get_default_graph())
    return _graph_replace(*args, **kwargs)

def extract_update_dict(update_ops):
    """Extract variables and their new values from Assign and AssignAdd ops.

    Args:
        update_ops: list of Assign and AssignAdd ops, typically computed using Keras' opt.get_updates()

    Returns:
        dict mapping from variable values to their updated value
    """
    name_to_var = {v.name: v for v in tf.global_variables()}
    updates = OrderedDict()
    for update in update_ops:
        var_name = update.op.inputs[0].name
        var = name_to_var[var_name]
        value = update.op.inputs[1]
        if update.op.type == 'Assign':
            updates[var.value()] = value
        elif update.op.type == 'AssignAdd':
            updates[var.value()] = var + value
        else:
            raise ValueError("Update op type (%s) must be of type Assign or AssignAdd"%update_op.op.type)
    return updates

def train_UNROLLEDGAN(data, g_net, d_net, name='UNROLLEDGAN',
                dim_z=128, n_iters=1e5, lr=1e-4, batch_size=128,unrolling_steps=5,beta1=0.5,epsilon=1e-8,
                sampler=sample_z, eval_funcs=[]):

    ### 0. Common preparation
    hyperparams = {'LR': lr}
    base_dir, out_dir, log_dir = create_dirs(name, g_net.name, d_net.name, hyperparams)

    tf.reset_default_graph()

    global_step = tf.Variable(0, trainable=False)
    increment_step = tf.assign_add(global_step, 1)
    lr = tf.constant(lr)


    ### 1. Define network structure
    x_shape = data.train.images[0].shape
    z0 = tf.placeholder(tf.float32, shape=[None, dim_z])            # Latent var.
    x0 = tf.placeholder(tf.float32, shape=(None,) + x_shape)        # Generated images

    G = g_net(z0, 'UNROLLEDGAN_G')
    D_real = d_net(x0, 'UNROLLEDGAN_D')
    D_fake = d_net(G, 'UNROLLEDGAN_D', reuse=True)

    #################
    #tf.reset_default_graph()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)) +tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))

    #loss = tf.reduce_mean(-tf.log(D_real)-tf.log(1-D_fake))
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "UNROLLEDGAN_G")
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "UNROLLEDGAN_D")
    # Vanilla discriminator update
    d_opt = Adam(lr=lr, beta_1=beta1, epsilon=epsilon)

    updates = d_opt.get_updates(disc_vars, [], loss)
    d_train_op = tf.group(*updates, name="d_train_op")

    # Unroll optimization of the discrimiantor
    if unrolling_steps > 0:
        # Get dictionary mapping from variables to their update value after one optimization step
        update_dict = extract_update_dict(updates)
        cur_update_dict = update_dict
        for i in xrange(unrolling_steps - 1):
            # Compute variable updates given the previous iteration's updated variable
            cur_update_dict = graph_replace(update_dict, cur_update_dict)
        # Final unrolled loss uses the parameters at the last time step
        unrolled_loss = graph_replace(loss, cur_update_dict)

    else:
        unrolled_loss = loss

    g_train_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, epsilon=epsilon)
    g_train_op = g_train_opt.minimize(-unrolled_loss, var_list=gen_vars)

    ################
    # Loss functions
    D_loss = tf.reduce_mean(-tf.log(D_real)-tf.log(1-D_fake))
    G_loss = tf.reduce_mean(-tf.log(D_fake))

    #D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
    #            .minimize(D_loss, var_list=get_trainable_params('UNROLLEDGAN_D'))
    #G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5))  \
    #            .minimize(G_loss, var_list=get_trainable_params('UNROLLEDGAN_G'))


    #### 2. Operations for log/state back-up
    tf.summary.scalar('UNROLLEDGAN_D(x)', tf.reduce_mean(D_real))
    tf.summary.scalar('UNROLLEDGAN_D(G)', tf.reduce_mean(D_fake))
    tf.summary.scalar('UNROLLEDGAN_D_loss', tf.reduce_mean(D_loss))
    tf.summary.scalar('UNROLLEDGAN_G_loss', tf.reduce_mean(G_loss))

    if check_dataset_type(x_shape) != 'synthetic':
        tf.summary.image('UNROLLEDGAN', G, max_outputs=4)        # for images only

    summaries = tf.summary.merge_all()

    saver = tf.train.Saver(get_trainable_params('UNROLLEDGAN_D') + get_trainable_params('UNROLLEDGAN_G'))

    # Initial setup for visualization
    outputs = [G]
    figs = [None] * len(outputs)
    fig_names = ['fig_gen_{:04d}_UNROLLEDGAN.png']

    plt.ion()

    ### 3. Run a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print('{:>10}, {:>7}, {:>7}, {:>7}') \
        .format('Iters', 'cur_LR', 'UNROLLEDGAN_D', 'UNROLLEDGAN_G')


    for it in range(int(n_iters)):
        batch_xs, batch_ys = data.train.next_batch(batch_size)
        f, _, _ = sess.run([[loss, unrolled_loss], g_train_op, d_train_op], feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)} )
        #_, loss_D = sess.run(
        #    [D_solver, D_loss],
        #    feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)}
        #)

        #_, loss_G = sess.run(
        #    [G_solver, G_loss],
        #    feed_dict={z0: sampler(batch_size, dim_z)}
        #)

        _, cur_lr = sess.run([increment_step, lr])

        if it % PRNT_INTERVAL == 0:
            print('{:10d}, {:1.4f}') \
                    .format(it, cur_lr)

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
            saver.save(sess, out_dir + 'UNROLLEDGAN', it)

    sess.close()


if __name__ == '__main__':
    args = parse_args(additional_args=[])
    print args

    if args.gpu:
        set_gpu(args.gpu)

    if args.datasets == 'mnist':
        out_name = 'UNROLLEDGAN_mnist'
        out_name = out_name if len(args.tag) == 0 else '{}_{}'.format(out_name, args.tag)

        dim_z = 64

        data = data_mnist.MnistWrapper('datasets/mnist/')
        g_net = SimpleGEN(dim_z, last_act=tf.sigmoid)
        d_net = SimpleCNN(1, last_act=tf.sigmoid)

        train_UNROLLEDGAN(data, g_net, d_net, name=out_name, dim_z=dim_z,  batch_size=args.batchsize, lr=args.lr,
                    eval_funcs=[lambda it, gen: eval_images_naive(it, gen, data)])

    elif args.datasets == 'celeba':
        out_name = 'UNROLLEDGAN_celeba'
        out_name = out_name if len(args.tag) == 0 else '{}_{}'.format(out_name, args.tag)

        dim_z = 128

        data = data_celeba.CelebA('datasets/img_align_celeba')
        g_net = UNROLLEDGAN_G(dim_z, last_act=tf.sigmoid)
        d_net = UNROLLEDGAN_D(1, last_act=tf.sigmoid)

        train_UNROLLEDGAN(data, g_net, d_net, name=out_name, dim_z=dim_z, batch_size=args.batchsize, lr=args.lr,
                    eval_funcs=[lambda it, gen: eval_images_naive(it, gen, data)])
