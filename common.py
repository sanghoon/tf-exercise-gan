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


# Global configs
DIM_Z = 128
BATCH_SIZE = 64
N_ITERS_PER_EPOCH = int(50000 / BATCH_SIZE)
N_ITERS = N_ITERS_PER_EPOCH * 200


# Helper functions
def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples, figId=None, retBytes=False):
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
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if retBytes:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return fig, buf

    return fig


def parse_args(modelnames=[]):
    parser = argparse.ArgumentParser()

    if len(modelnames) > 0:
        parser.add_argument('-net', choices=modelnames, default=None)

    parser.add_argument('-w_clip', type=float, default=0.1)
    parser.add_argument('-bn', action='store_true', default=False)
    parser.add_argument('-nobn', action='store_true', default=False)        # FIXME
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-tag', type=str, default='')
    parser.add_argument('-kernel', type=int, default=5)                     # only for ConvNets

    args = parser.parse_args()

    if args.nobn == True:
        assert(args.bn is not True)

    return args
