# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.utilities.util_common import look_up_operations
from Demic.net.pnet import PNet

def fuse_layer_w_initializer():
    def _initializer(shape, dtype, partition_info):
        assert(shape[0]==3)
        w_init0 = np.random.rand(shape[1], shape[2], shape[3], shape[4])*1e-5
        w_init2 = np.random.rand(shape[1], shape[2], shape[3], shape[4])*1e-5
        w_init1 = 1 - w_init0 - w_init2
        w_init = np.asarray([w_init0, w_init1, w_init2])
        w_init = tf.constant(w_init, tf.float32)
        return w_init
    return _initializer

class PNet_Multi_Slice(TrainableLayer):
    """
        PNet_Multi_Slice
        The input tensor shape is [N, D, H, W, C]
        network parameters:
        -- num_slices： number of slices for fusion
        -- fuse_features: number of output features for fuse layer
        -- num_features: features for P-Net, default [64, 64, 64, 64, 64]
        -- dilations:    dilation of P-Net, default [1, 2, 3, 4, 5]
        """
    
    def __init__(self,
                 num_classes,
                 parameters = None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='PNet_Multi_Slice'):
        super(PNet_Multi_Slice, self).__init__(name=name)
        self.parameters = parameters
        self.acti_func = acti_func
        self.num_classes = num_classes
        self.num_slices  = parameters['num_slices']
        self.num_features = parameters.get('num_features', 16)
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        fuse_layer = ConvolutionalLayer(n_output_chns=self.num_features,
                                kernel_size=[self.num_slices,1,1],
                                w_initializer = fuse_layer_w_initializer(),
                                w_regularizer = self.regularizers['w'],
                                padding = 'Valid',
                                name='fuse_layer')
        pnet_layer = PNet(self.num_classes,
                          parameters = self.parameters,
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name = 'pnet_layer')
        output = fuse_layer(images, is_training, bn_momentum)
        output = pnet_layer(output, is_training, bn_momentum)
        return output
