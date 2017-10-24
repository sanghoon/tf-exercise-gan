from common import *
from datasets import data_celeba, data_mnist
from models.celeba_models import *
from models.mnist_models import *
from eval_funcs import *

def log(x):
    return tf.log(x + 1e-8)


def train_modegan(data, g_net, d_enc, d_net, name='MODEGAN',
                 dim_z=128, n_iters=1e5, lr=1e-4, batch_size=16,
                 sampler=sample_z, eval_funcs=[],
                 l_k=0.001, g_k=0.5):
    ### 0. Common preparation
    hyperparams = {'LR': lr}
    base_dir, out_dir, log_dir = create_dirs(name, g_net.name, d_enc.name, hyperparams)

    tf.reset_default_graph()

    global_step = tf.Variable(0, trainable=False)
    increment_step = tf.assign_add(global_step, 1)
    lr = tf.constant(lr)

    ### 1. Define network structure
    x_shape = data.train.images[0].shape
    z0 = tf.placeholder(tf.float32, shape=[None, dim_z])            # Latent var.
    x0 = tf.placeholder(tf.float32, shape=(None,) + x_shape)        # Generated images

    G_sample = g_net(z0,'MODEGAN_G')
    G_sample_reg = g_net(d_enc(x0,'MODEGAN_E'),'MODEGAN_G',reuse=True)

    D_real = d_net(x0,'MODEGAN_D')
    D_fake = d_net(G_sample,'MODEGAN_D',reuse=True)
    D_reg = d_net(G_sample_reg,'MODEGAN_D',reuse=True)

    mse = tf.reduce_sum((x0 - G_sample_reg)**2, 1)
    lam1 = 1e-2
    lam2 = 1e-2

    D_loss = -tf.reduce_mean(log(D_real) + log(1 - D_fake))
    E_loss = tf.reduce_mean(lam1 * mse + lam2 * log(D_reg))
    G_loss = -tf.reduce_mean(log(D_fake)) + E_loss

    E_solver = (tf.train.AdamOptimizer(learning_rate=lr)
                .minimize(E_loss, var_list=get_trainable_params('MODEGAN_E')))
    D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
                .minimize(D_loss, var_list=get_trainable_params('MODEGAN_D')))
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
                .minimize(G_loss, var_list=get_trainable_params('MODEGAN_G')))

#operations for log/state back-up
    tf.summary.scalar('MODEGAN_D_loss', D_loss)
    tf.summary.scalar('MODEGAN_G_loss', G_loss)
    tf.summary.scalar('MODEGAN_E_loss', E_loss)

    if check_dataset_type(x_shape) != 'synthetic':
        tf.summary.image('MODEGAN', G, max_outputs=4)        # for images only

    summaries = tf.summary.merge_all()

    saver = tf.train.Saver(get_trainable_params('MODEGAN_D') + get_trainable_params('MODEGAN_G') + get_trainable_params('MODEGAN_E') )

    # Initial setup for visualization
    outputs = [G_sample_reg, G_sample]
    figs = [None] * len(outputs)
    fig_names = ['fig_gen_{:04d}_MODEGAN.png', 'fig_gen_reg_{:04d}_MODEGAN.png']

    plt.ion()


    ### 3. Run a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print('{:>10}, {:>7}, {:>7}, {:>7}, {:>7}') \
        .format('Iters', 'cur_LR', 'MODEGAN_D', 'MODEGAN_G','MODEGAN_E')


    for it in range(int(n_iters)):
        batch_xs, batch_ys = data.train.next_batch(batch_size)
        _, loss_D = sess.run(
            [D_solver, D_loss],
            feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)}
        )

        _, loss_G = sess.run(
            [G_solver, G_loss],
            feed_dict={x0:batch_xs, z0: sampler(batch_size, dim_z)}
        )
        _, loss_E = sess.run(
            [E_solver, E_loss],
            feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)}
        )


        _, cur_lr = sess.run([increment_step, lr])

        if it % PRNT_INTERVAL == 0:
            print('{:10d}, {:1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}') \
                    .format(it, cur_lr, loss_D, loss_G, loss_E)

            # Tensorboard
            cur_summary = sess.run(summaries, feed_dict={x0: batch_xs, z0: sampler(batch_size, dim_z)})
            writer.add_summary(cur_summary, it)

        #if it % EVAL_INTERVAL == 0:
        if it % 500 == 0: #TODO
            # FIXME
            img_generator = lambda n: sess.run(output, feed_dict={x0:batch_xs,z0: sampler(n, dim_z)})

            for i, output in enumerate(outputs):
                figs[i] = data.plot(img_generator, fig_id=i)
                figs[i].canvas.draw()
                plt.savefig(out_dir + fig_names[i].format(it / 1000), bbox_inches='tight')

            # Run evaluation functions
            for func in eval_funcs:
                func(it, img_generator)

        if it % SAVE_INTERVAL == 0:
            saver.save(sess, out_dir + 'modegan', it)

    sess.close()

