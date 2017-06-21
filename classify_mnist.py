import os
# if DISPLAY is not defined
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')       # Use a different backend
import tensorflow as tf
from utils import *
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from models import *


# Load dataset
data = input_data.read_data_sets('data/mnist/', one_hot=True)

trainimg = data.train.images
testimg = data.test.images

print '# of training data: {}'.format(data.train.num_examples)
print 'type of train. img: {}'.format(type(data.train.images))
print 'type of train. lbl: {}'.format(type(data.train.labels))
print 'shape of trai. img: {}'.format(data.train.images.shape)
print 'shape of trai. lbl: {}'.format(data.train.labels.shape)


def augment_image(im):
    im = tf.image.random_flip_left_right(im)
    return im

x0 = tf.placeholder(tf.float32, shape=[None, 784])
x1 = tf.reshape(x0, [-1, 28, 28, 1])
x2 = tf.map_fn(augment_image, x1)                       # data augmentation
y0 = tf.placeholder(tf.float32, shape=[None, 10])       # GT


# Parameters
lr = 0.001          #   fixed lr
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.01, global_step, 500, 0.96, staircase=True)   # *0.96 per each epoch
training_epochs = 200
batch_size = 100

out = simple_cnn(x2, n_out=10, is_training=True, last_act=tf.identity)
y_pred = tf.nn.softmax(out)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y0))
optm = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

correct_pred = tf.equal(tf.argmax(y0, 1), tf.argmax(y_pred, 1))
accu = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#
tst_out = simple_cnn(x1, n_out=10, is_training=False, reuse=True)
tst_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tst_out, labels=y0))
tst_pred = tf.nn.softmax(tst_out)
tst_corr = tf.equal(tf.argmax(y0, 1), tf.argmax(tst_pred, 1))
tst_accu = tf.reduce_mean(tf.cast(tst_corr, tf.float32))



# Let's start
sess = tf.Session()
sess.run(tf.initialize_all_variables())

print('{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}') \
    .format('Iters', 'cur_LR', 'loss', 'accu.', 'tstloss', 'tstaccu')

for it in xrange(training_epochs * 500):
    batch_xs, batch_ys = data.train.next_batch(batch_size)  # Sample a batch
    sess.run(optm, feed_dict={x0: batch_xs, y0: batch_ys})  # Run optimizer

    if it % 100 == 0:
        cur_lr = sess.run(lr)

        cur_loss, cur_accu = sess.run([loss, accu], feed_dict={x0: batch_xs, y0: batch_ys})

        batch_tst_xs, batch_tst_ys = data.test.next_batch(10000)        # full batch
        cur_tstloss, cur_tstaccu = sess.run([tst_loss, tst_accu], feed_dict={x0: batch_tst_xs, y0: batch_tst_ys})

        print '{:7d}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}, {: 1.4f}'   \
                .format(it / 100, cur_lr, cur_loss, cur_accu, cur_tstloss, cur_tstaccu)

