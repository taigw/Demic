# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.utilities.util_common import look_up_operations
from net.pnet import PNet
class PNet_Multi_Slice(TrainableLayer):
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
                 acti_func='prelu',
                 name='PNet_Multi_Slice'):
        super(PNet_Multi_Slice, self).__init__(name=name)
        
        self.acti_func = acti_func
        self.num_classes = num_classes
        
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        pnet_layer = PNet(self.num_classes,
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name = 'pnet_layer')
        # transpose the input from [N, D, H, W, 1] to [N, 1, H, W, D]
        output = tf.transpose(images, perm=[0,4,2,3,1])
        output = pnet_layer(output, is_training, bn_momentum)
        return output
