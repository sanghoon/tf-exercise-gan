from common import plot
import os.path
import glob
import cv2
import random
import numpy as np


class ImgDataset:
    def __init__(self, dataDir, i_from=0, i_to=None, shuffle=False, crop=None, resize=None):
        self.dataDir = dataDir
        self.img_list = glob.glob(os.path.join(dataDir, "*.jpg"))
        self.img_list = sorted(self.img_list)[i_from:i_to]

        self.shuffle = shuffle
        self._i = 0

        self.resize = resize        # Expected width and height
        self.crop = crop            # Cropsize at center

        if shuffle:
            random.shuffle(self.img_list)

        self.preloaded = False
        self.images = [self[0]]     # Dummy image for size calculation in other codes

    def preload(self):
        images = map(cv2.imread, self.img_list)
        images = map(self.crop_and_resize, images)

        self.images = images
        self.preloaded = True

    def crop_and_resize(self, im):
        # Crop
        if self.crop:
            h, w, = im.shape[:2]
            j = int(round((h - self.crop) / 2.))
            i = int(round((w - self.crop) / 2.))

            im = im[j:j+self.crop, i:i+self.crop, :]

        if self.resize:
            im = cv2.resize(im, (self.resize, self.resize))

        # rescale (range: 0.0~1.0)
        im = im.astype(np.float32) / 255.0

        # Reverse RGB-ordering (Please refer http://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/)
        im = im[:, :, (2,1,0)]

        return im

    def __getitem__(self, item):
        if self.preloaded:
            return self.images[item]

        if isinstance(item, tuple) or isinstance(item, slice):
            im = map(cv2.imread, self.img_list[item])
            im = map(self.crop_and_resize, im)
        else:
            # Read image
            im = cv2.imread(self.img_list[item])
            im = self.crop_and_resize(im)

        return im

    def __len__(self):
        return len(self.img_list)

    def next_batch(self, batch_size):
        samples = self[self._i : self._i + batch_size]
        self._i += batch_size

        # If reached the end of the dataset
        if self._i >= len(self):
            # Re-initialize
            self._i = 0
            if self.shuffle:
                random.shuffle(self.img_list)

            n_more = batch_size - len(samples)
            samples = samples + self.next_batch(n_more)[0]

        return samples, None


class CelebA:
    def __init__(self, dataDir):
        self.train = ImgDataset(dataDir, i_from=0, i_to=150000, shuffle=True, crop=108, resize=64)
        self.validation = ImgDataset(dataDir, i_from=150000, i_to=160000, crop=108, resize=64)
        self.test = ImgDataset(dataDir, i_from=160000, i_to=None, crop=108, resize=64)

        # for evaluation
        self.validation.preload()

        # TODO: Follow the original set's train/val/test pratition
        # TODO: Provide label info.

    # TODO: refactoring
    def plot(self, img_generator, fig_id=None):
        samples = img_generator(16)
        fig = plot(samples, fig_id, shape=self.train.images[0].shape)

        return fig


if __name__ == '__main__':
    import sys

    dataDir = sys.argv[1]
    data = CelebA(dataDir)

    ims, _ = data.train.next_batch(16)

    for i in range(16):
        cv2.imshow('image', ims[i][:, :, (2,1,0)])
        cv2.waitKey(0)

    ims, _ = data.test.next_batch(16)

    for i in range(16):
        cv2.imshow('image', ims[i][:, :, (2,1,0)])
        cv2.waitKey(0)