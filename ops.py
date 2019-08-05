from __future__ import division, print_function, absolute_import
import tensorflow as tf
from functools import partial


def linear(input, output_dim, scope='linear', stddev=0.01):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('biases', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def residual_block(input, output_dim, k=3, s=2, stddev=0.02, bn=None, bn_input=True,
                   activation_fn=None, atv_input=True, resample=None, fstr=False, name="block"):
    input_shape = input.get_shape().as_list()
    input_dim = input_shape[-1]
    new_height = input_shape[1] * s
    new_width = input_shape[2] * s
    norm = tf.random_normal_initializer(stddev=stddev)
    he_initializer = tf.contrib.layers.variance_scaling_initializer()
    conv2d = partial(tf.layers.conv2d, padding="SAME")
    conv2d_he = partial(tf.layers.conv2d, kernel_initializer=he_initializer, padding="SAME")
    resize = partial(tf.image.resize_nearest_neighbor, size=(new_height, new_width))
    mean_pool = partial(tf.nn.avg_pool, ksize=[1, 2, 2, 1], strides=[1, s, s, 1], padding="SAME")
    conv2d_transpose = partial(tf.layers.conv2d_transpose, kernel_size=k, strides=s, kernel_initializer=norm,
                               padding="SAME")

    if resample == "up":
        # skip connection
        skip = input
        if input_dim != output_dim:
            skip = conv2d(skip, filters=output_dim, kernel_size=1, strides=1, name=name + ".skip.conv")
        skip = resize(skip, name=name + ".skip")

        # first convolutional layer
        h = input
        if bn is not None and bn_input:
            h = bn(h, scope=name + ".conv1.bn")
        if activation_fn is not None and atv_input:
            h = activation_fn(h, name=name + ".conv1.act")
        h = conv2d_he(h, filters=output_dim, kernel_size=k, strides=1, name=name + ".conv1.lin")

        # second convolutional layer
        if bn is not None:
            h = bn(h, scope=name + ".conv2.bn")
        if activation_fn is not None:
            h = activation_fn(h, name=name + ".conv2.act")

        if not fstr:
            h = resize(h, name=name + ".conv2.resize")
            h = conv2d_he(h, filters=output_dim, kernel_size=k, strides=1, name=name + ".conv2.lin")
        else:
            h = conv2d_transpose(h, filters=output_dim, name=name + ".conv2_transpose.lin")

        output = tf.add(h, skip, name=name + ".output")
    elif resample == "down":
        # skip connection
        skip = input
        if input_dim != output_dim:
            skip = conv2d(skip, filters=output_dim, kernel_size=1, strides=1, name=name + ".skip.conv")
        skip = mean_pool(skip, name=name + ".skip")

        # first convolutional layer
        h = input
        if bn is not None and bn_input:
            h = bn(h, scope=name + ".conv1.bn")
        if activation_fn is not None and atv_input:
            h = activation_fn(h, name=name + ".conv1.act")
        h = conv2d_he(h, filters=output_dim, kernel_size=k, strides=1, name=name + ".conv1.lin")

        # second convolutional layer
        if bn is not None:
            h = bn(h, scope=name + ".conv2.bn")
        if activation_fn is not None:
            h = activation_fn(h, name=name + ".conv2.act")

        if not fstr:
            h = conv2d_he(h, filters=output_dim, kernel_size=k, strides=1, name=name + ".conv2.lin")
            h = mean_pool(h, name=name + ".conv2.mean_pool")
        else:
            h = conv2d_he(h, filters=output_dim, kernel_size=k, strides=2, name=name + ".conv2.lin")

        output = tf.add(h, skip, name=name + ".output")
    else:
        # skip connection
        skip = input
        if input_dim != output_dim:
            skip = conv2d(skip, filters=output_dim, kernel_size=1, strides=1, name=name + ".skip.conv")

        # fist convolutional layer
        h = input
        if bn is not None and bn_input:
            h = bn(h, scope=name + ".conv1.bn")
        if activation_fn is not None and atv_input:
            h = activation_fn(h, name=name + ".conv1.act")
        h = conv2d_he(h, filters=output_dim, kernel_size=k, strides=1, name=name + ".conv1.lin")

        # second convolutional layer
        if bn is not None:
            h = bn(h, scope=name + ".conv2.bn")
        if activation_fn is not None:
            h = activation_fn(h, name=name + ".conv2.act")
        h = conv2d_he(h, filters=output_dim, kernel_size=k, strides=1, name=name + ".conv2.lin")

        output = tf.add(h, skip, name=name + ".output")

    return output
