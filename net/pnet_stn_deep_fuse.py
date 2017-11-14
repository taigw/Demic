# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer,
from niftynet.layer.fully_connected import ConvLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.utilities.util_common import look_up_operations
from net.pnet import PNet
from net.pnet_stn import MultiSliceSpatialTransform


class PNet_STN_DF(TrainableLayer):
    """
        Reimplementation of P-Net
        Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." MICCAI 2015
        The input tensor shape is [N, D, H, W, C] where D is 1
        """
    
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 input_shape = [5, 3, 96, 96, 1],
                 num_features = 16,
                 acti_func='prelu',
                 name='PNet_STN_DF'):
        super(PNet_STN_DF, self).__init__(name=name)
        
        self.acti_func = acti_func
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.num_features = num_features
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        stn_layer = MultiSliceSpatialTransform(self.input_shape,
                                               w_initializer = self.initializers['w'],
                                               w_regularizer = self.regularizers['w'],
                                               name = 'stn_layer')
        pnet_layer = PNet(self.num_classes,
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name = 'pnet_layer')
        fuse_layer = ConvLayer(n_output_chns=self.num_classes,
                             kernel_size=[self.num_slices,1,1],
                             w_initializer = fuse_layer_w_initializer(),
                             w_regularizer = self.regularizers['w'],
                             padding = 'Valid',
                             name='fuse_layer')
        output = stn_layer(images, is_training, bn_momentum)
        output = pnet_layer(output, is_training, bn_momentum)
        output = fuse_layer(output)
        return output

if __name__ == '__main__':
    batch_size = 5
    input_shape = [batch_size,3,96,96,1]
    x = tf.placeholder(tf.float32, shape = input_shape)
    net = PNet_STN(2, input_shape = input_shape)
    y = net(x, is_training=True)
    print(y)
