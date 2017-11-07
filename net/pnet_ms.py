# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from net.pnet import PNet

class PNet_MS(TrainableLayer):
    """
        P-Net with multiple slice. The neighouring slices are used as multi-channels
        input tensor shape is [N, D, H, W, C] where D is 1
        """
    
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='PNet_MS'):
        super(PNet_MS, self).__init__(name=name)

        self.pnet = PNet(num_classes,
                         w_initializer,
                         w_regularizer,
                         b_initializer,
                         b_regularizer,
                         acti_func)
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, layer_id=-1):
        images_transpose = tf.transpose(images, [0, 4, 2, 3, 1])
        output = self.pnet(images_transpose)
        return output
