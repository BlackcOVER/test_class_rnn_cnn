import tensorflow as tf
import tensorflow.contrib as tf_contrib
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import *

# tf.nn   ops
# tf.layers  layer


def variable_placeholer(size, name):
    '''
    :param size: size of variable
    :param name: name of variable
    :return: variable
    '''
    return tf.placeholder(shape=size, name=name)


def embedding_layer(size, name, x, trainable=True, init_w=None):
    '''
    :param size: size of embedding
    :param name: name of embedding ops
    :param x: input x
    :param trainable: trainable or no
    :param init_w: pretrained_vectors
    :return: ops of embedding_layer
    '''
    with tf.variable_scope('embedding_layer'):
        if init_w:
            W = tf.get_variable(name=name, shape=size, initializer=tf.constant_initializer(init_w), trainable=trainable)
        else:
            W = tf.get_variable(name=name, shape=size, trainable=trainable)
        emdeded = tf.nn.embedding_lookup(W, x)
        return emdeded


# softmax层
def soft_max_layer(logits):
    '''
    :param logits:
    :return: 预测label
    '''
    with tf.variable_scope('soft_max_layer'):
        predict_y = tf.arg_max(tf.nn.softmax(logits=logits), 1)
    return predict_y


# 初始化变量
def get_variable(name, shape, trainable=True, initializer=None):
    variable = tf.get_variable(name, shape, tf.float32, trainable=trainable, initializer=initializer)
    return variable


# 2d卷积层
# padding = "SAME" 矩阵大小不变（补0）  = "VALID" 不补0，size = size - 2
def conv2(input, name, source_dim, dim, kernel_size, reuse=None, padding='SAME'):
    with tf.variable_scope(name, reuse=reuse):
        kernel = get_variable('weight', [kernel_size, kernel_size, source_dim, dim], initializer=tf.random_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding=padding)
        bias = get_variable('bias', [dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias)
        return conv


# 残差模块
def res_block(x, name):
    res_conv1 = conv2(x, '{}_conv1'.format(name), 64, 64, 3)
    activate = tf.nn.relu(res_conv1, name='{}_act1'.format(name))
    res_conv2 = conv2(activate, '{}_conv2'.format(name), 64, 64, 3)
    res_logit = res_conv2 + x
    output = tf.nn.relu(res_logit, name='{}_act2'.format(name))
    return output




# batch_normal层
def bn(x, use_bias=False, bn_decay=0.9997):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if use_bias:
        bias = get_variable('bias', params_shape, initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = get_variable('gamma', params_shape, initializer=tf.ones_initializer)
    moving_mean = get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_variance = get_variable('moving_variance', params_shape, initializer=tf.ones_initializer, trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)

    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, bn_decay)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, bn_decay)
    # 向当前计算图中添加张量
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(True, lambda: (mean, variance), lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    return x