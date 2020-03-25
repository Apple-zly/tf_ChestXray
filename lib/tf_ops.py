
'''
author: kiclent@yahoo.com
Tensorflow 1.7.1
'''

import numpy as np
import tensorflow as tf

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        k = kernel[0]
        c = input.get_shape().as_list()[-1]
        std = np.sqrt(2/k/k/c)
        network = tf.layers.conv2d(inputs=input,
                                   use_bias=False,
                                   filters=filter,
                                   kernel_size=kernel,
                                   strides=stride,
                                   kernel_initializer=tf.random_normal_initializer(stddev=std),
                                   padding=padding)
        return network

def Fully_connected(x, units, layer_name='fully_connected') :
    std = np.sqrt(2/(1+x.get_shape().as_list()[-1])/units)
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x,
                               use_bias=True,
                               units=units,
                               kernel_initializer=tf.random_normal_initializer(stddev=std),
                               bias_initializer=tf.random_normal_initializer(stddev=std),
                               activation=None)

def Leaky_relu(x):
    return tf.nn.leaky_relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Relu(x):
    return tf.nn.relu(x)

def Softmax(x):
    return tf.nn.softmax(x)

def Global_Average_Pooling(x):
    N, H, W, C = x.get_shape().as_list()
    return tf.layers.average_pooling2d(x, (H, W), (1, 1), padding='VALID', name='Global_avg_pooling')

def Max_pooling(x, pool_size=(3,3), stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding, name='max_pooling')

def Avg_pooling(x, pool_size=(3,3), stride=2, padding='SAME') :
    return tf.layers.average_pooling2d(x, pool_size=pool_size, strides=stride, padding=padding, name='avg_pooling')

def Concatenation(layers):
    return tf.concat(layers, axis=3)

def Dropout(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Flatten(x):
    return tf.layers.flatten(x)

# 批量归一化
def Batch_Normalization(x, scope, training, epsilon=1e-3, decay=0.9):

    def bn_layer(x, scope, training, epsilon=1e-3, decay=0.9, reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            shape = x.get_shape().as_list()
            gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
            beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
            moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0),
                                         trainable=False)
            if training:
                avg, var = tf.nn.moments(x, np.arange(len(shape) - 1), keep_dims=True)
                avg = tf.reshape(avg, [avg.shape.as_list()[-1]])
                var = tf.reshape(var, [var.shape.as_list()[-1]])
                update_moving_avg = tf.assign(moving_avg, moving_avg * decay + avg * (1 - decay))
                update_moving_var = tf.assign(moving_var, moving_var * decay + var * (1 - decay))
                control_inputs = [update_moving_avg, update_moving_var]
            else:
                avg = moving_avg
                var = moving_var
                control_inputs = []
            with tf.control_dependencies(control_inputs):
                output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

        return output

    return tf.cond(
        training,
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, training=True, reuse=None),
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, training=False, reuse=True),
    )

