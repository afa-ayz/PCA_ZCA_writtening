from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import urllib
import scipy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

DATA_DIRECTORY = "data/"
LOGS_DIRECTORY = "logs/train"

# train params
training_epochs = 30
batch_size = 100
display_step = 50

# network params
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

# Store layers weight & bias

with tf.name_scope('weight'):
    normal_weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='w1_normal'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='w2_normal'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='wout_normal')
    }

with tf.name_scope('bias'):
    normal_biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1_normal'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2_normal'),
        'out': tf.Variable(tf.random_normal([n_classes]), name='bout_normal')
    }
    zero_biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1]), name='b1_zero'),
        'b2': tf.Variable(tf.zeros([n_hidden_2]), name='b2_zero'),
        'out': tf.Variable(tf.zeros([n_classes]), name='bout_normal')
    }

weight_initializer = {'normal': normal_weights}
bias_initializer = {'normal': normal_biases, 'zero': zero_biases}

# user input
from argparse import ArgumentParser

WEIGHT_INIT = 'normal'
BIAS_INIT = 'normal'
BACH_NORM = True


def pca_whitening(x):
    e = 1e-5
    cov = np.dot(x.T, x) / x.shape[0]
    U, S, V = np.linalg.svd(cov)
    xRot = np.dot(x, U.T)
    xPCAwhite = xRot * (np.diag(1. / np.sqrt(np.diag(S) + e)))
    return xPCAwhite


def zcawhitening(x):
    e = 1e-5
    cov = tf.matmul(tf.transpose(x), x) / x.shape[0]
    U, S, V = np.linalg.svd(cov)
    xRot = tf.matmul(x, tf.transpose(U))
    xPCAwhite = xRot * (tf.diag(1. / tf.sqrt(tf.diag(S) + e)))
    zca = xPCAwhite.dot(U)
    return zca


def cholosky_whitening(inputT):
    # This fucntion ONLY works on more or equal than 2 rank
    cov, _ = tf.contrib.metrics.streaming_covariance(inputT[0], input[1])
    L = tf.cholesky(cov)
    output = tf.sqrt(
        tf.matmul(
            tf.transpose(
                tf.matmul(
                    tf.transpose(L), inputs)), (tf.matmul(
                        tf.transpose(L), inputs))))
    return output


def batch_norm_FOR_CHECK(inputT, is_training=True, scope=None):

    return tf.cond(
        is_training,
        lambda: batch_norm(
            inputT,
            is_training=True,
            center=True,
            scale=True,
            activation_fn=tf.nn.relu,
            decay=0.9,
            scope=scope),
        lambda: batch_norm(
            inputT,
            is_training=False,
            center=True,
            scale=True,
            activation_fn=tf.nn.relu,
            decay=0.9,
            scope=scope,
            reuse=True))


def batch_norm_with_PCA_whitening(inputs, is_training=True):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    shift = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    test_mean = tf.Variable(
        tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    test_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        inputs = tf.divide(inputs, tf.sqrt(batch_var))
        return tf.nn.batch_normalization(
            inputs, batch_mean, batch_var, shift, scale, 1e-3)
    else:
        return tf.nn.batch_normalization(
            inputs, test_mean, test_var, shift, scale, 1e-3)


def batch_norm_with_CHO_whitening(
        inputs,
        is_training=True,
        decay=0.5,
        scope=None):
    inputs = tf.reshape(inputs, ([16, 16]))
    print(inputs.shape)
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    shift = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    test_mean = tf.Variable(
        tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    test_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        # batch_mean, batch_var = tf.nn.moments(inputs, [0])
        inputs = cholosky_whitening(inputs)
        return tf.nn.batch_normalization(
            inputs, batch_mean, batch_var, shift, scale, 1e-3)
    else:
        return tf.nn.batch_normalization(
            inputs, test_mean, test_var, shift, scale, 1e-3)


def batch_norm_with_decay_whitening(
        inputs,
        is_training=True,
        decay=0.5,
        scope=None):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    shift = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    test_mean = tf.Variable(
        tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    test_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(test_mean,
                               test_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(test_var,
                              test_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                inputs, batch_mean, batch_var, shift, scale, 1e-3)
    else:
        return tf.nn.batch_normalization(
            inputs, test_mean, test_var, shift, scale, 1e-3)

# Create model of MLP with batch-normalization layer


def MLPwithBN(x, weights, biases, Train=True):
    with tf.name_scope('MLPwithBN'):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = batch_norm_with_decay_whitening(layer_1, is_training=Train)
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = batch_norm_with_decay_whitening(layer_2, is_training=Train)
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Create model of MLP without batch-normalization layer


def MLPwoBN(x, weights, biases):
    with tf.name_scope('Cholesky whitening on BN'):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# main function


def main():

    weights = weight_initializer['normal']
    biases = bias_initializer['normal']
    batch_normalization = True
    # Import data
    mnist = input_data.read_data_sets('data/', one_hot=True)

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])  # answer

    # Predict

    y = MLPwithBN(x, weights, biases, is_training)

    # Get loss of model
    with tf.name_scope("LOSS"):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=y, labels=y_))

    # Define optimizer
    with tf.name_scope("ADAM"):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # moving_mean and moving_variance need to be updated
    if batch_normalization == "True":
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            train_ops = [train_step] + update_ops
            train_op_final = tf.group(*train_ops)
        else:
            train_op_final = train_step
    else:
        train_op_final = train_step
    # Get accuracy of model
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor loss tensor
    tf.summary.scalar('loss', loss)

    # Create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Training cycle
    total_batch = int(mnist.train.num_examples / batch_size)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(
        LOGS_DIRECTORY, graph=tf.get_default_graph())

    # Loop for epoch
    for epoch in range(training_epochs):

        # Loop over all batches
        for i in range(total_batch):

            batch = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            _, train_accuracy, summary = sess.run([train_op_final, accuracy, merged_summary_op], feed_dict={
                                                  x: batch[0], y_: batch[1], is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print(
                    "Epoch:", '%04d,' %
                    (epoch + 1), "batch_index %4d/%4d, training accuracy %.5f" %
                    (i, total_batch, train_accuracy))

    # Calculate accuracy for all mnist test images
    print(
        "test accuracy for the latest result: %g" %
        accuracy.eval(
            feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels,
                is_training: False}))


if __name__ == '__main__':
    main()
