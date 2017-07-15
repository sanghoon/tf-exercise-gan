from tensorflow.examples.tutorials.mnist import input_data
from common import plot

class MnistWrapper:
    def __init__(self, datapath='datasets/mnist/'):
        self.dataset = input_data.read_data_sets(datapath, one_hot=True, reshape=False)

        self.train = self.dataset.train
        self.validation = self.dataset.validation
        self.test = self.dataset.test

    # TODO: refactoring
    def plot(self, img_generator, fig_id=None):
        samples = img_generator(16)
        fig = plot(samples, fig_id, shape=self.train.images[0].shape)

        return fig