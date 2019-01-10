# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.crop import CropLayer
from niftynet.utilities.util_common import look_up_operations

class DenseUNet2D5MultiScale(TrainableLayer):
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
                 name='DenseUNet2D5MultiScale'):
        super(DenseUNet2D5MultiScale, self).__init__(name=name)
        
        if(parameters is None):
            self.n_features = [32, 64, 96, 128, 160]
            self.dropout    = 0.8
            self.margin     = 0
        else:
            self.n_features = parameters.get('num_features', [32, 64, 96, 128, 160])
            self.dropout    = parameters.get('dropout', 0.8)
            self.margin     = parameters.get('margin', 0)
        
        self.acti_func = acti_func
        self.num_classes = num_classes
        
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        # image_size  should be divisible by 8
#        spatial_dims = images.get_shape()[1:-1].as_list()
#        assert (spatial_dims[-2] % 16 == 0 )
#        assert (spatial_dims[-1] % 16 == 0 )
        
        block1 = DenseUNetBlock(self.n_features[0],
                            (1,3,3),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B1')
                            
        block2 = DenseUNetBlock(self.n_features[1],
                            (1,3,3), 
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B2')
                            
        block3 = DenseUNetBlock(self.n_features[2],
                            (1,3,3),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B3')
                            
        block4 = DenseUNetBlock(self.n_features[3], 
                            (3,3,3),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B4')
                            
        block5 = DenseUNetBlock(self.n_features[4], 
                           (3,3,3), 
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           acti_func=self.acti_func,
                           name='B5')
            
        block6 = DenseUNetBlock(self.n_features[3], 
                          (1,3,3),
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name='B6')

        block7 = DenseUNetBlock(self.n_features[2],
                           (1,3,3), 
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           acti_func=self.acti_func,
                           name='B7')
                           
        block8 = DenseUNetBlock(self.n_features[1], 
                          (1,3,3), 
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name='B8')
                          
        block9 = DenseUNetBlock(self.n_features[0], 
                         (1,3,3), 
                         w_initializer=self.initializers['w'],
                         w_regularizer=self.regularizers['w'],
                         acti_func=self.acti_func,
                         name='B9')
                         
        conv1 = ConvLayer(n_output_chns=self.num_classes,
                               kernel_size=(1,1,1),
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               with_bias = True,
                               name='conv1')
        conv2 = ConvLayer(n_output_chns=self.num_classes,
                                kernel_size=(1,1,1),
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                with_bias = True,
                                name='conv2')
        conv3 = ConvLayer(n_output_chns=self.num_classes,
                                kernel_size=(1,1,1),
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                with_bias = True,
                                name='conv3')
        conv4 = ConvLayer(n_output_chns=self.num_classes,
                                kernel_size=(1,1,1),
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                with_bias = True,
                                name='conv4')
        down1 = DownSampleLayer('MAX', kernel_size=(1,2,2), stride=(1,2,2), name='down1')
        down2 = DownSampleLayer('MAX', kernel_size=(1,2,2), stride=(1,2,2), name='down2')
        down3 = DownSampleLayer('MAX', kernel_size=(1,2,2), stride=(1,2,2), name='down3')
        down4 = DownSampleLayer('MAX', kernel_size=(2,2,2), stride=(2,2,2), name='down4')
        
        up1 = DeconvolutionalLayer(n_output_chns=self.n_features[3], kernel_size=(2,2,2), stride=(2,2,2), name='up1')
        up2 = DeconvolutionalLayer(n_output_chns=self.n_features[2], kernel_size=(1,2,2), stride=(1,2,2), name='up2')
        up3 = DeconvolutionalLayer(n_output_chns=self.n_features[1], kernel_size=(1,2,2), stride=(1,2,2), name='up3')
        up4 = DeconvolutionalLayer(n_output_chns=self.n_features[0], kernel_size=(1,2,2), stride=(1,2,2), name='up4')
        
        up_f7_1 = DeconvolutionalLayer(n_output_chns=self.num_classes, kernel_size=(1,2,2), stride=(1,2,2), name='up4_f7_1')
        up_f7_2 = DeconvolutionalLayer(n_output_chns=self.num_classes, kernel_size=(1,2,2), stride=(1,2,2), name='up4_f7_2')
        up_f8 =   DeconvolutionalLayer(n_output_chns=self.num_classes, kernel_size=(1,2,2), stride=(1,2,2), name='up4_f8')

        centra_slice = TensorSliceLayer(margin = self.margin)
        f1 = block1(images, is_training, bn_momentum)
        d1 = down1(f1)
        f2 = block2(d1, is_training, bn_momentum)
        d2 = down2(f2)
        f3 = block3(d2, is_training, bn_momentum)
        d3 = down3(f3)
        f4 = block4(d3, is_training, bn_momentum)
        d4 = down4(f4)
        f5 = block5(d4, is_training, bn_momentum)
        # add dropout to the original version
        f5 = tf.nn.dropout(f5, self.dropout)
        
        f5up = up1(f5, is_training, bn_momentum)
        f4cat = tf.concat((f4, f5up), axis = -1)
        f6 = block6(f4cat, is_training, bn_momentum)
        # add dropout to the original version
        f6 = tf.nn.dropout(f6, self.dropout)

        f6up = up2(f6, is_training, bn_momentum)
        f3cat = tf.concat((f3, f6up), axis = -1)
        f7 = block7(f3cat, is_training, bn_momentum)
        # add dropout to the original version
        f7 = tf.nn.dropout(f7, self.dropout)
        f7_pred = conv1(f7)
        f7_pred = up_f7_1(f7_pred, is_training, bn_momentum)
        f7_pred = up_f7_2(f7_pred, is_training, bn_momentum)

        f7up = up3(f7, is_training, bn_momentum)
        f2cat = tf.concat((f2, f7up), axis = -1)
        f8 = block8(f2cat, is_training, bn_momentum)
        # add dropout to the original version
        f8 = tf.nn.dropout(f8, self.dropout)
        f8_pred = conv2(f8)
        f8_pred = up_f8(f8_pred, is_training, bn_momentum)

        f8up = up4(f8, is_training, bn_momentum)
        f1cat = tf.concat((f1, f8up), axis = -1)
        f9 = block9(f1cat, is_training, bn_momentum)
        # add dropout to the original version
        f9 = tf.nn.dropout(f9, self.dropout)
        f9_pred = conv3(f9)

        pred_cat = tf.concat((f7_pred, f8_pred, f9_pred), axis = -1)
        output = conv4(pred_cat)
        if(self.margin > 0):
            output = centra_slice(output)
        return output


SUPPORTED_OP = {'DOWNSAMPLE', 'UPSAMPLE', 'NONE'}


class DenseUNetBlock(TrainableLayer):
    def __init__(self,
                 n_chn,
                 kernel,
                 w_initializer=None,
                 w_regularizer=None,
                 acti_func='relu',
                 name='DenseUNet_block'):
        
        super(DenseUNetBlock, self).__init__(name=name)
        
        self.kernel = kernel
        self.n_chn = n_chn
        self.acti_func = acti_func
        
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}
    
    def layer_op(self, input_tensor, is_training, bn_momentum = 0.9):
        conv_op1 = ConvolutionalLayer(n_output_chns= self.n_chn,
                                         kernel_size=self.kernel,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=self.acti_func,
                                         name='{}_1'.format(self.n_chn))

        conv_op2 = ConvolutionalLayer(n_output_chns= self.n_chn,
                                         kernel_size=self.kernel,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=self.acti_func,
                                         name='{}_2'.format(self.n_chn))
        
        conv_op3 = ConvolutionalLayer(n_output_chns= self.n_chn,
                                         kernel_size=self.kernel,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=self.acti_func,
                                         name='{}_3'.format(self.n_chn))

        f1 =  conv_op1(input_tensor, is_training, bn_momentum)
        f1cat =  tf.concat((input_tensor, f1), axis = -1)
        f2 =  conv_op2(f1cat, is_training, bn_momentum)
        f2cat =  tf.concat((input_tensor, f1, f2), axis = -1)
        f3 =  conv_op3(f2cat, is_training, bn_momentum)
    
        return f3

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