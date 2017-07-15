from common import plot
import matplotlib.pyplot as plt
import random
import numpy as np
from attrdict import AttrDict
import tensorflow.contrib.learn as tf_learn


class MoG:
    def __init__(self):
        self.modes = []

        # Mimic tf datasets generator
        self.train = AttrDict({'next_batch':
                                   lambda b: (self.next_batch(b), None)})

        self.train.images = [np.zeros([2])]

    def add_mode(self, x, y, std=1.0):
        x = float(x)
        y = float(y)
        std = float(std)

        self.modes.append({'x': x,'y': y, 'std': std})

        return self

    def generate_sample(self, with_label=False):
        # Pick a mode
        mode = random.choice(self.modes)

        x = np.random.normal(mode['x'], mode['std'])
        y = np.random.normal(mode['y'], mode['std'])

        return (x,y)

    def estimate_mode_idx(self, x, y, thres=3.0):
        x = float(x)
        y = float(y)
        thres = float(thres)

        _min_dist = np.inf
        _min_i = -1

        for i, mode in enumerate(self.modes):
            m_x = mode['x']
            m_y = mode['y']
            m_std = mode['std']

            dist = np.sqrt((m_x - x) * (m_x - x) + (m_y - y) * (m_y - y))

            if (dist <= thres * m_std):
                # Keep the index with minimum dist.
                if (dist < _min_dist):
                    _min_i = i

        return _min_i

    def estimate_mode_idxs(self, arr, thres=3.0):
        ret = np.apply_along_axis(lambda x:
                                    self.estimate_mode_idx(x[0], x[1], thres),
                                  1,
                                  arr
                                  )

        return ret

    def next_batch(self, batchsize=128):
        numbers = []

        for i in range(batchsize):
            numbers.append(self.generate_sample())

        return np.array(numbers)

    def get_hq_ratio(self, arr, thres=3.0):
        ret = self.estimate_mode_idxs(arr, thres)

        return np.sum(ret >= 0) / float(len(ret))

    def get_n_modes(self, arr, thres=3.0):
        visited = [False for x in self.modes]

        ret = self.estimate_mode_idxs(arr, thres)
        for r in ret:
            if r >= 0:
                visited[r] = True

        return sum(visited)

    # TODO: refactoring
    def plot(self, img_generator, fig_id=None):
        samples = img_generator(1024)
        fig = plot(samples, fig_id, shape=self.train.images[0].shape)

        # Plot true samples
        modes = [(m['x'], m['y']) for m in self.modes]
        modes = np.array(modes)

        plt.figure(fig_id)
        plt.scatter(modes[:, 0], modes[:, 1])

        return fig

    @property
    def n_modes(self):
        return len(self.modes)


class Spiral():
    # Please refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/synthetic.py
    def __init__(self, size=4, std=0.05):
        self.train = AttrDict({'next_batch': self.next_batch})

        self.train.images = [np.zeros([2])]

        self.std = std
        self.size = size

        self.n_modes = 100

    def next_batch(self, n_samples):
        X, Y = tf_learn.datasets.synthetic.spirals(n_samples, self.std, n_loops=1)
        X = self.size * X

        return X, Y

    def get_hq_ratio(self, samples, thres=3.0):
        # TODO: Count # of samples within threshold dist.

        true_X, _ = tf_learn.datasets.synthetic.spirals(self.n_modes, n_loops=1)
        true_X *= self.size

        # Naive impl.
        dist = np.zeros([self.n_modes, len(samples)], dtype=np.float)
        for i in range(self.n_modes):
            for j in range(len(samples)):
                dist[i, j] = np.linalg.norm(true_X[i] - samples[j])

        hq_cnt = np.sum(np.min(dist, axis=0) < self.std * thres * self.size)

        return hq_cnt / float(len(samples))


    def get_n_modes(self, arr, thres=3.0):
        true_X, _ = tf_learn.datasets.synthetic.spirals(self.n_modes, n_loops=1)
        true_X *= self.size

        # Naive impl.
        dist = np.zeros([self.n_modes, len(arr)], dtype=np.float)
        for i in range(self.n_modes):
            for j in range(len(arr)):
                dist[i, j] = np.linalg.norm(true_X[i] - arr[j])

        visited = np.any((dist < self.std * thres * self.size), axis=1)

        return sum(visited)

    # TODO: refactoring
    def plot(self, img_generator, fig_id=None):
        samples = img_generator(1024)
        fig = plot(samples, fig_id, shape=self.train.images[0].shape)

        # Plot true samples
        true_X, _ = tf_learn.datasets.synthetic.spirals(self.n_modes, n_loops=1)
        true_X *= self.size

        plt.figure(fig_id)
        plt.scatter(true_X[:,0], true_X[:, 1])

        return fig


def rect_MoG(size, std=0.25):
    assert(size % 2 == 1)

    mog = MoG()

    _start = - size + 1
    _end = size
    _std = std

    for i in range(_start, _end, 2):
        for j in range(_start, _end, 2):
            mog.add_mode(i, j, _std)

    return mog


if __name__ == '__main__':
    # Create
    mog = rect_MoG(5)

    # datasets = mog.generate_batch(4096)
    data = mog.train.next_batch(4096)
    plt.scatter(data[0][:,0], data[0][:,1], alpha=0.1)

    data = Spiral().train.next_batch(4096)
    plt.scatter(data[0][:, 0], data[0][:, 1], alpha=0.1)


    plt.show()

