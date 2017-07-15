#!/usr/bin/env python

from collections import OrderedDict
from datasets.data_synthetic import *
from dcgan import *
from madgan import *
from began import *
from models.toy_models import *


def eval_synthetic(it, samples, data):
    metrics = OrderedDict()

    # Simple metric for MoG (VEEGAN, https://arxiv.org/abs/1705.07761)
    if isinstance(data, MoG) or isinstance(data, Spiral):
        metrics['hq_ratio'] = data.get_hq_ratio(samples) * 100.0
        metrics['modes_ratio'] = data.get_n_modes(samples) / float(data.n_modes) * 100.0

    print "Eval({}) ".format(it), ', '.join(['{}={:.2f}'.format(k, v) for k, v in metrics.iteritems()])


if __name__ == '__main__':
    args = parse_args(additional_args=[
        ('--n_gen', {'type': int, 'default': 8})
    ,])
    print args

    if args.gpu:
        set_gpu(args.gpu)

    dim_z = 64
    dim_x = 2

    n_generators = args.n_gen

    data = rect_MoG(5)

    # Evaluation func.
    true_samples, _ = data.train.next_batch(1024)

    eval_func = lambda it, sample_generator:\
                    eval_synthetic(it, sample_generator(1024), data)

    # MADGAN
    g_net = ToyNet(dim_x, dim_z, dim_h=128, last_act=tf.identity, act=tf.nn.elu, bn=False)
    d_net = ToyNet(n_generators + 1, dim_x, dim_h=128, last_act=tf.identity, act=leaky_relu, bn=False)

    train_madgan(data, g_net, d_net, dim_z=dim_z, n_generators=n_generators, batch_size=256, lr=1e-4,
                 eval_funcs=[eval_func])
