# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.utilities.util_common import look_up_operations
from net.pnet import PNet

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

class PNet_Multi_Slice_Deep_Fuse(TrainableLayer):
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
                 num_slices = 3,
                 num_features = 16,
                 acti_func='prelu',
                 name='PNet_Multi_Slice_Deep_Fuse'):
        super(PNet_Multi_Slice_Deep_Fuse, self).__init__(name=name)
        
        self.acti_func = acti_func
        self.num_classes = num_classes
        self.num_slices = num_slices
        self.num_features = num_features
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
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
        output = pnet_layer(images, is_training, bn_momentum)
        output = fuse_layer(output)
        
        return output
