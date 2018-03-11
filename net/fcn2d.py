# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer

from niftynet.utilities.util_common import look_up_operations

class FCN2D(TrainableLayer):
    """
        FCN2D
            The input tensor shape is [N, D, H, W, C] , where D is 1
            network parameters:
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
                 name='FCN2D'):
        super(FCN2D, self).__init__(name=name)
        if(parameters is None):
            self.n_features = [64, 128, 256, 512, 1024]
            self.dropout = 0.8
        else:
            self.n_features = parameters.get('num_features', [64, 128, 256, 512, 1024])
            self.dropout    = parameters.get('dropout', 0.8)
        self.acti_func = acti_func
        self.num_classes = num_classes
        
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        block1 = FCNBlock((self.n_features[0], self.n_features[0]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B1')
                            
        block2 = FCNBlock((self.n_features[1], self.n_features[1]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B2')
                            
        block3 = FCNBlock((self.n_features[2],self.n_features[2],self.n_features[2]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B3')
                            
        block4 = FCNBlock((self.n_features[3],self.n_features[3],self.n_features[3]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B4')
                            
        block5 = FCNBlock((self.n_features[4],self.n_features[4],self.n_features[4]),
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           acti_func=self.acti_func,
                           name='B5')
        
        conv6 = ConvolutionalLayer(n_output_chns=self.n_features[-1]*2,
                                     kernel_size=[1,3,3],
                                     with_bias = True,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     acti_func=self.acti_func,
                                     name='conv6')
                                     
        conv7 = ConvolutionalLayer(n_output_chns=self.n_features[-1]*2,
                                  kernel_size=[1,1,1],
                                  with_bias = True,
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  acti_func=self.acti_func,
                                  name='conv7')
                                  
        conv_score3 = ConvolutionalLayer(n_output_chns=self.num_classes,
                                         kernel_size=[1,3,3],
                                         with_bias = True,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func = None,
                                         name='score3')

        conv_score4 = ConvolutionalLayer(n_output_chns=self.num_classes,
                                         kernel_size=[1,3,3],
                                         with_bias = True,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func = None,
                                         name='score4')
                                         
        conv_score5 = ConvolutionalLayer(n_output_chns=self.num_classes,
                                         kernel_size=[1,3,3],
                                         with_bias = True,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func = None,
                                         name='score5')
                                 
        up1 = DeconvolutionalLayer(n_output_chns=self.num_classes, kernel_size=(1,4,4), stride=(1,2,2),
                                   with_bias = True, acti_func = None, name='up1')
        up2 = DeconvolutionalLayer(n_output_chns=self.num_classes, kernel_size=(1,4,4), stride=(1,2,2),
                                   with_bias = True, acti_func = None, name='up2')
        up3 = DeconvolutionalLayer(n_output_chns=self.num_classes, kernel_size=(1,16,16), stride=(1,8,8),
                                   with_bias = True, acti_func = None, name='up3')
        
        f1 = block1(images, is_training, bn_momentum)
        f2 = block2(f1, is_training, bn_momentum)
        f3 = block3(f2, is_training, bn_momentum)
        f3 = tf.nn.dropout(f3, self.dropout)
        f4 = block4(f3, is_training, bn_momentum)
        f4 = tf.nn.dropout(f4, self.dropout)
        f5 = block5(f4, is_training, bn_momentum)
        f5 = tf.nn.dropout(f5, self.dropout)
        f6 = conv6(f5, is_training, bn_momentum)
        f6 = tf.nn.dropout(f6, self.dropout)
        f7 = conv7(f6, is_training, bn_momentum)
        f7 = tf.nn.dropout(f7, self.dropout)
        score3 = conv_score3(f3, is_training, bn_momentum)
        score4 = conv_score4(f4, is_training, bn_momentum)
        score5 = conv_score5(f7, is_training, bn_momentum)
        
        pred = up1(score5, is_training, bn_momentum)
        pred = pred + score4
        pred = up2(pred, is_training, bn_momentum)
        pred = pred + score3
        pred = up3(pred, is_training, bn_momentum)
        return pred

class FCNBlock(TrainableLayer):
    def __init__(self,
                 n_chns,
                 w_initializer=None,
                 w_regularizer=None,
                 acti_func='relu',
                 name='FCN_block'):
        
        super(FCNBlock, self).__init__(name=name)
        
        self.n_chns = n_chns
        self.acti_func = acti_func
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}
    
    def layer_op(self, input_tensor, is_training, bn_momentum):
        output_tensor = input_tensor
        for i in range(len(self.n_chns)):
            conv_op = ConvolutionalLayer(n_output_chns = self.n_chns[i],
                                         kernel_size   = [1,3,3],
                                         with_bias     = True,
                                         w_initializer = self.initializers['w'],
                                         w_regularizer = self.regularizers['w'],
                                         acti_func     = self.acti_func,
                                         name='{}'.format(i))
            output_tensor = conv_op(output_tensor, is_training, bn_momentum)
        pooling = DownSampleLayer('MAX', kernel_size=(1,2,2), stride=(1,2,2), name='down1')
        output_tensor = pooling(output_tensor)
        return output_tensor
