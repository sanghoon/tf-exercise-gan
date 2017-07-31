#!/usr/bin/env python
from datasets.data_synthetic import *
from dcgan import *
from madgan import *
from began import *
from wgan import *
from gogan import *
from models.toy_models import *
from eval_funcs import eval_synthetic


if __name__ == '__main__':
    args = parse_args(lr=1e-4,
                      additional_args=[('--n_gen', {'type': int, 'default': 8}),])
    print args

    if args.gpu:
        set_gpu(args.gpu)

    # Network params
    dim_x = 2
    dim_h = 128
    dim_z = 64
    dim_ae = 32     # BEGAN only
    n_generators = args.n_gen   # MADGAN only

    # Training params
    params = {'n_iters': 100001, 'batch_size': args.batchsize, 'lr': args.lr, 'dim_z': dim_z}

    # Output storage
    stat_entries = {}

    for name, data in [('SynMoG', rect_MoG(5)), ('SynSpiral', Spiral())]:
        # Evaluation func.
        gen_eval_func = lambda tag: \
            lambda it, sample_generator: eval_synthetic(it, sample_generator, data, tag=tag)

        # Common generator
        g_net = ToyNet(dim_x, dim_z, dim_h=dim_h, last_act=tf.identity, act=tf.nn.elu, bn=False)

        # Disc. for DCGAN (sigmoid)
        d_net = ToyNet(1, dim_x, dim_h=dim_h, last_act=tf.sigmoid, act=leaky_relu, bn=False)
        train_dcgan(data, g_net, d_net, name='DCGAN_' + name,
                    eval_funcs=[gen_eval_func('DCGAN_' + name)],
                    **params)

        # Disc. for MADGAN (multi-output)
        d_net = ToyNet(n_generators + 1, dim_x, dim_h=dim_h, last_act=tf.identity, act=leaky_relu, bn=False)
        train_madgan(data, g_net, d_net, name='MADGAN_' + name, n_generators=n_generators,
                     eval_funcs=[gen_eval_func('MADGAN_' + name)],
                     **params)

        # Disc. for WGAN and GoGAN (identity)
        d_net = ToyNet(1, dim_x, dim_h=dim_h, last_act=tf.identity, act=leaky_relu, bn=False)
        train_wgan(data, g_net, d_net, name='WGAN_' + name,
                   eval_funcs=[gen_eval_func('WGAN_' + name)],
                   **params)
        train_gogan(data, g_net, d_net, name='GoGAN_' + name,
                   eval_funcs=[gen_eval_func('GoGAN_' + name)],
                   **params)

        # Encoder-decoder for BEGAN
        d_enc = ToyNet(dim_ae, dim_x, dim_h=dim_h, last_act=tf.identity, act=leaky_relu, bn=False)
        d_dec = ToyNet(dim_x, dim_ae, dim_h=dim_h, last_act=tf.identity, act=tf.nn.elu, bn=False)

        train_began(data, g_net, d_enc, d_dec, name='BEGAN_' + name,
                    eval_funcs=[gen_eval_func('BEGAN_' + name)],
                    **params)