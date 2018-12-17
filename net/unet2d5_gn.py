# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.crop import CropLayer
from Demic.layer.convolution import ConvLayer, ConvolutionalLayer

class UNet2D5GN(TrainableLayer):
    """
        Reimplementation of U-Net
        Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." MICCAI 2015
        The input tensor shape is [N, D, H, W, C] where D is 1
        """
    
    def __init__(self,
                 num_classes,
                 parameters   =None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='UNet2D5GN'):
        super(UNet2D5GN, self).__init__(name=name)
        
        if(parameters is None):
            self.n_features = [16, 32, 64, 96, 128]
            self.dropout    = 0.8
            self.margin     = 0
        else:
            self.n_features = parameters.get('num_features', [16, 32, 64, 96, 128])
            self.dropout    = parameters.get('dropout', 0.8)
            self.margin     = parameters.get('margin', 0)
        self.acti_func = acti_func
        self.num_classes = num_classes
        
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))

    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        # image_size  should be divisible by 8
        
        block1 = UNetBlock((self.n_features[0], self.n_features[0]),
                            ((1,3,3), (1,3,3)),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B1')
                            
        block2 = UNetBlock((self.n_features[1], self.n_features[1]),
                            ((1,3,3), (1,3,3)),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B2')
                            
        block3 = UNetBlock((self.n_features[2], self.n_features[2]),
                            ((1,3,3), (1,3,3)),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B3')
                            
        block4 = UNetBlock((self.n_features[3], self.n_features[3]),
                            ((3,3,3), (3,3,3)),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B4')
                            
        block5 = UNetBlock((self.n_features[4], self.n_features[4]),
                           ((3,3,3), (3,3,3)),
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           acti_func=self.acti_func,
                           name='B5')
            
        block6 = UNetBlock((self.n_features[3], self.n_features[3]),
                          ((1,3,3), (1,3,3)),
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name='B6')

        block7 = UNetBlock((self.n_features[2], self.n_features[2]),
                           ((1,3,3), (1,3,3)),
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           acti_func=self.acti_func,
                           name='B7')
                           
        block8 = UNetBlock((self.n_features[1], self.n_features[1]),
                          ((1,3,3), (1,3,3)),
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name='B8')
                          
        block9 = UNetBlock((self.n_features[0], self.n_features[0]),
                         ((1,3,3), (1,3,3)),
                         w_initializer=self.initializers['w'],
                         w_regularizer=self.regularizers['w'],
                         acti_func=self.acti_func,
                         name='B9')
                         
        conv = ConvLayer(n_output_chns=self.num_classes,
                               kernel_size=(1,1,1),
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               with_bias = True,
                               name='conv')
        down1 = DownSampleLayer('MAX', kernel_size=(1,2,2), stride=(1,2,2), name='down1')
        down2 = DownSampleLayer('MAX', kernel_size=(1,2,2), stride=(1,2,2), name='down2')
        down3 = DownSampleLayer('MAX', kernel_size=(1,2,2), stride=(1,2,2), name='down3')
        down4 = DownSampleLayer('MAX', kernel_size=(2,2,2), stride=(2,2,2), name='down4')
        
        up1 = DeconvolutionalLayer(n_output_chns=self.n_features[3], kernel_size=(2,2,2), stride=(2,2,2), name='up1')
        up2 = DeconvolutionalLayer(n_output_chns=self.n_features[2], kernel_size=(1,2,2), stride=(1,2,2), name='up2')
        up3 = DeconvolutionalLayer(n_output_chns=self.n_features[1], kernel_size=(1,2,2), stride=(1,2,2), name='up3')
        up4 = DeconvolutionalLayer(n_output_chns=self.n_features[0], kernel_size=(1,2,2), stride=(1,2,2), name='up4')
        
        centra_slice = TensorSliceLayer(margin = self.margin)
        f1 = block1(images, is_training)
        d1 = down1(f1)
        f2 = block2(d1, is_training)
        d2 = down2(f2)
        f3 = block3(d2, is_training)
        d3 = down3(f3)
        f4 = block4(d3, is_training)
        d4 = down4(f4)
        f5 = block5(d4, is_training)
        # add dropout to the original version
        f5 = tf.nn.dropout(f5, self.dropout)
        
        f5up = up1(f5, is_training)
        f4cat = tf.concat((f4, f5up), axis = -1)
        f6 = block6(f4cat, is_training)
        # add dropout to the original version
        f6 = tf.nn.dropout(f6, self.dropout)

        f6up = up2(f6, is_training)
        f3cat = tf.concat((f3, f6up), axis = -1)
        f7 = block7(f3cat, is_training)
        # add dropout to the original version
        f7 = tf.nn.dropout(f7, self.dropout)

        f7up = up3(f7, is_training)
        f2cat = tf.concat((f2, f7up), axis = -1)
        f8 = block8(f2cat, is_training)
        # add dropout to the original version
        f8 = tf.nn.dropout(f8, self.dropout)

        f8up = up4(f8, is_training)
        f1cat = tf.concat((f1, f8up), axis = -1)
        f9 = block9(f1cat, is_training)
        # add dropout to the original version
        f9 = tf.nn.dropout(f9, self.dropout)
        output = conv(f9)

        if(self.margin > 0):
            output = centra_slice(output)
        return output


SUPPORTED_OP = {'DOWNSAMPLE', 'UPSAMPLE', 'NONE'}


class UNetBlock(TrainableLayer):
    def __init__(self,
                 n_chns,
                 kernels,
                 w_initializer=None,
                 w_regularizer=None,
                 acti_func='relu',
                 name='UNet_block'):
        
        super(UNetBlock, self).__init__(name=name)
        
        self.kernels = kernels
        self.n_chns = n_chns
        self.acti_func = acti_func
        
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}
    
    def layer_op(self, input_tensor, is_training):
        output_tensor = input_tensor
        for (kernel_size, n_features) in zip(self.kernels, self.n_chns):
            conv_op = ConvolutionalLayer(n_output_chns=n_features,
                                         kernel_size=kernel_size,
                                         with_bias = True,
                                         with_bn = False,
                                         group_size = 32,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=self.acti_func,
                                         name='{}'.format(n_features))
            output_tensor = conv_op(output_tensor, is_training)
    
        return output_tensor


class TensorSliceLayer(TrainableLayer):
    """
    extract the central part of a tensor
    """

    def __init__(self, margin = 1, regularizer=None, name='tensor_extract'):
        self.layer_name = name
        super(TensorSliceLayer, self).__init__(name=self.layer_name)
        self.margin = margin
        
    def layer_op(self, input_tensor):
        input_n      = input_tensor.get_shape().as_list()[1]
        idx_range    = tf.constant(np.asarray(range(self.margin, input_n - self.margin)))
        input_trans  = tf.transpose(input_tensor, [1, 0, 2, 3, 4])
        output_trans = tf.gather(input_trans, idx_range)
        output = tf.transpose(output_trans, [1, 0, 2, 3, 4])
        return output
