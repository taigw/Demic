# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.utilities.util_common import look_up_operations
from Demic.net.pnet import PNet
from Demic.net.pnet_stn import MultiSliceSpatialTransform

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

class PNet_STN_DF(TrainableLayer):
    """
        PNet_STN_DF
        The input tensor shape is [N, D, H, W, C]
        network parameters:
        -- input_shape：input shape of network, e.g. [5, 3, 96, 96, 1]
        -- num_features: features for P-Net, default [64, 64, 64, 64, 64]
        -- dilations:    dilation of P-Net, default [1, 2, 3, 4, 5]

        """
    
    def __init__(self,
                 num_classes,
                 parameters   =None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='PNet_STN_DF'):
        super(PNet_STN_DF, self).__init__(name=name)
        self.parameters = parameters
        self.acti_func = acti_func
        self.num_classes = num_classes
        self.input_shape = parameters['input_shape']
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        stn_layer = MultiSliceSpatialTransform(self.input_shape,
                                               w_initializer = self.initializers['w'],
                                               w_regularizer = self.regularizers['w'],
                                               name = 'stn_layer')
        pnet_layer = PNet(self.num_classes,
                          self.parameters,
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name = 'pnet_layer')
                          
        fuse_layer = ConvLayer(n_output_chns=self.num_classes,
                             kernel_size=[self.input_shape[1],1,1],
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
