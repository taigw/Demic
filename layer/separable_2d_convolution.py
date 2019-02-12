# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.utilities.util_common import look_up_operations
from Demic.layer.normalization import BNLayer, GNLayer


SUPPORTED_PADDING = set(['SAME', 'VALID'])


def default_w_initializer():
    def _initializer(shape, dtype, partition_info):
        stddev = np.sqrt(2.0 / np.prod(shape[:-1]))
        from tensorflow.python.ops import random_ops
        return random_ops.truncated_normal(shape, 0.0, stddev, dtype=tf.float32)
        # return tf.truncated_normal_initializer(
        #    mean=0.0, stddev=stddev, dtype=tf.float32)

    return _initializer


def default_b_initializer():
    return tf.constant_initializer(0.0)


class SeparableConv2DLayer(TrainableLayer):
    """
    This class defines a 2d convolutional layer with depthwise convolution
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='separable_conv_2d'):
        super(SeparableConv2DLayer, self).__init__(name=name)

        self.padding = look_up_operations(padding.upper(), SUPPORTED_PADDING)
        self.n_output_chns = int(n_output_chns)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.with_bias = with_bias

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor):
        # the input tensor should have a shape of [N, C, H, W]
        input_shape = input_tensor.shape.as_list()
        n_input_chns = input_shape[-1]
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)
        assert(len(input_shape) == 4)
        assert(spatial_rank == 2)
        # initialize conv kernels/strides and then apply
        w_full_size = layer_util.expand_spatial_params(
            self.kernel_size, spatial_rank)
        # expand kernel size to include number of features
        w_full_size = w_full_size + (n_input_chns, 1)#self.n_output_chns)
        full_stride = layer_util.expand_spatial_params(
            self.stride, spatial_rank)
        full_dilation = layer_util.expand_spatial_params(
            self.dilation, spatial_rank)

        # channelwise convolution
        conv_kernel = tf.get_variable(
            'w', shape=w_full_size,
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])
        output_tensor = tf.nn.depthwise_conv2d(input=input_tensor,
                                          filter=conv_kernel,
                                          strides=full_stride,
                                          rate=full_dilation,
                                          padding=self.padding,
                                          name='channelwise_conv')
        # inter-channel convolution
        conv_kernel2 = tf.get_variable(
            'w', shape=[1, 1, n_input_chns, self.n_output_chns],
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])

        output_tensor = tf.nn.convolution(input = output_tensor,
                                        filter = conv_kernel2,
                                        padding = self.padding,
                                        name = 'interchannel_conv')
        if not self.with_bias:
            return output_tensor

        # adding the bias term
        bias_term = tf.get_variable(
            'b', shape=self.n_output_chns,
            initializer=self.initializers['b'],
            regularizer=self.regularizers['b'])
        output_tensor = tf.nn.bias_add(output_tensor,
                                       bias_term,
                                       name='add_bias')
        return output_tensor


class SeparableConvolutional2DLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components::

        convolution -> batch_norm -> activation -> dropout

    The b_initializer and b_regularizer are applied to the ConvLayer
    The w_initializer and w_regularizer are applied to the ConvLayer,
    the batch normalisation layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 with_bn=True,
                 group_size=-1,
                 acti_func=None,
                 preactivation=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name="separable_convolution_2d"):

        self.acti_func = acti_func
        self.with_bn = with_bn
        self.group_size = group_size
        self.preactivation = preactivation
        self.layer_name = '{}'.format(name)
        if self.with_bn and group_size > 0:
            raise ValueError('only choose either batchnorm or groupnorm')
        if self.with_bn:
            self.layer_name += '_bn'
        if self.group_size > 0:
            self.layer_name += '_gn'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(SeparableConvolutional2DLayer, self).__init__(name=self.layer_name)

        # for ConvLayer
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.with_bias = with_bias

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        conv_layer = SeperableConv2DLayer(n_output_chns=self.n_output_chns,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               dilation=self.dilation,
                               padding=self.padding,
                               with_bias=self.with_bias,
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               b_initializer=self.initializers['b'],
                               b_regularizer=self.regularizers['b'],
                               name='conv_')

        if self.with_bn:
            if is_training is None:
                raise ValueError('is_training argument should be '
                                 'True or False unless with_bn is False')
            bn_layer = BNLayer(
                regularizer=self.regularizers['w'],
                moving_decay=self.moving_decay,
                eps=self.eps,
                name='bn_')
        if self.group_size > 0:
            gn_layer = GNLayer(
                regularizer=self.regularizers['w'],
                eps=self.eps,
                name='gn_')
        if self.acti_func is not None:
            acti_layer = ActiLayer(
                func=self.acti_func,
                regularizer=self.regularizers['w'],
                name='acti_')

        if keep_prob is not None:
            dropout_layer = ActiLayer(func='dropout', name='dropout_')

        def activation(output_tensor):
            if self.with_bn:
                output_tensor = bn_layer(output_tensor, is_training)
            if self.group_size > 0:
                output_tensor = gn_layer(output_tensor, self.group_size, is_training)
            if self.acti_func is not None:
                output_tensor = acti_layer(output_tensor)
            if keep_prob is not None:
                output_tensor = dropout_layer(output_tensor,
                                              keep_prob=keep_prob)
            return output_tensor

        if self.preactivation:
            output_tensor = conv_layer(activation(input_tensor))
        else:
            output_tensor = activation(conv_layer(input_tensor))

        return output_tensor