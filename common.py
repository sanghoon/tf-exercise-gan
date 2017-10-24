import os
# if DISPLAY is not defined
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')       # Use a different backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse
import numpy as np
import io
import matplotlib.cm as cm

# Global configs
PRNT_INTERVAL = 100
EVAL_INTERVAL = 2000
SHOW_FIG_INTERVAL = 100
SAVE_INTERVAL = 4000

DATASETS = ['mnist', 'celeba']


# Helper functions
def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])



def create_dirs(name, g_name, d_name, hyperparams=None):
    base_dir = 'out/{}_{}_{}/'.format(name, g_name, d_name) \
                + '_'.join(['{}={}'.format(k,v) for (k,v) in hyperparams.iteritems()])

    out_dir = os.path.join(base_dir, 'out/')
    log_dir = os.path.join(base_dir, 'log/')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return base_dir, out_dir, log_dir



# Naive impl. with pre-defined numbers
def check_dataset_type(shape):
    assert(shape)

    if len(shape) == 1:
        return 'synthetic'
    elif shape[2] == 1:
        assert(shape[0] == 28 and shape[1] == 28)
        return 'mnist'
    elif shape[2] == 3:
        return 'celeba'

    return None


def plot(samples, figId=None, retBytes=False, shape=None):
    if figId is None:
        fig = plt.figure(figsize=(4, 4))
    else:
        fig = plt.figure(figId, figsize=(4,4))

    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if shape and shape[2] == 3:
            rescaled = np.clip(sample, 0.0, 1.0)
            plt.imshow(rescaled.reshape(*shape))
        else:
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if retBytes:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return fig, buf

    return fig


def scatter(samples, figId=None, retBytes=False, xlim=None, ylim=None):
    if figId is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figId)
        fig.clear()
    
    nGen = 8 #TODO
    colors = cm.rainbow(np.linspace(0, 1, nGen)) #TODO
    colors = np.repeat(colors, len(samples[:,0])/nGen, 0) #TODO
    
    #plt.scatter(samples[:,0], samples[:,1], c = colors, alpha=0.1) #TODO
    plt.scatter(samples[:,0], samples[:,1], alpha=0.1)

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if retBytes:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return fig, buf

    return fig



def parse_args(batchsize=128, lr=1e-5, additional_args=[]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0)   #TODO
    parser.add_argument('--batchsize', type=int, default=batchsize)
    parser.add_argument('--datasets', choices=DATASETS, default=DATASETS[0])
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--tag', type=str, default='')

    for key, kwargs in additional_args:
        parser.add_argument(key, **kwargs)

    args = parser.parse_args()

    return args


def set_gpu(gpu_id):
    print "Override GPU setting: gpu={}".format(gpu_id)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
