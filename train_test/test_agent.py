"""Script for testing
Author: Guotai Wang
"""

import os
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.contrib.data import Iterator
from Demic.net.net_factory import NetFactory
from Demic.image_io.image_loader import ImageLoader
from Demic.image_io.file_read_write import *
from Demic.image_io.image_loader import ImageLoader
from Demic.util.parse_config import parse_config
from Demic.util.image_process import resize_ND_volume_to_given_shape

def extract_roi_from_nd_volume(volume, roi_center, roi_shape, fill = 'random'):
    '''Extract an roi from a nD volume
        volume      : input nD numpy array
        roi_center  : center of roi with position
        output_shape: shape of roi
        '''
    input_shape = volume.shape
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = roi_shape)
    else:
        output = np.zeros(roi_shape)
    r0max = [int(x/2) for x in roi_shape]
    r1max = [roi_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], roi_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - roi_center[i]) for i in range(len(r0max))]
    out_center = r0max
    if(len(roi_center)==3):
        output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                      range(out_center[1] - r0[1], out_center[1] + r1[1]),
                      range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                    range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                    range(roi_center[2] - r0[2], roi_center[2] + r1[2]))]
    elif(len(roi_center)==4):
        output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                      range(out_center[1] - r0[1], out_center[1] + r1[1]),
                      range(out_center[2] - r0[2], out_center[2] + r1[2]),
                      range(out_center[3] - r0[3], out_center[3] + r1[3]))] = \
        volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                      range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                      range(roi_center[2] - r0[2], roi_center[2] + r1[2]),
                      range(roi_center[3] - r0[3], roi_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output

def set_roi_to_nd_volume(volume, roi_center, sub_volume):
    '''Set an roi of an ND volume with a sub volume
        volume: an ND numpy array
        roi_center: center of roi
        sub_volume: the sub volume that will be copied
        '''
    volume_shape = volume.shape
    patch_shape  = sub_volume.shape
    output_volume = volume
    for i in range(len(roi_center)):
        if(roi_center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], roi_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - roi_center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(roi_center) == 3):
        output_volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                             range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                             range(roi_center[2] - r0[2], roi_center[2] + r1[2]))] = \
        sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                          range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                          range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(roi_center) == 4):
        output_volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                             range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                             range(roi_center[2] - r0[2], roi_center[2] + r1[2]),
                             range(roi_center[3] - r0[3], roi_center[3] + r1[3]))] = \
        sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                          range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                          range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                          range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output_volume

def volume_probability_prediction_3d_roi(img, data_shape, label_shape,
                                         class_num, batch_size, sess, x, proby):
    '''
        Test one image with sub regions along x, y, z axis
        img        : a 4D numpy array with shape [D, H, W, C]
        data_shape : input 4d tensor shape
        label_shape: output 4d tensor shape
        class_num  : number of output class
        batch_size : batch size for testing
        sess       : tensorflow session that can run a graph
        x          : input tensor of the graph
        proby      : output tensor of the graph
        '''
    [D, H, W, C] = img.shape
    prob = np.zeros([D, H, W, class_num], np.float32)
    sub_image_patches = []
    sub_image_centers = []
    roid_half = int(label_shape[0]/2)
    roih_half = int(label_shape[1]/2)
    roiw_half = int(label_shape[2]/2)
    
    # get image patches
    for centerd in range(roid_half, D + roid_half, label_shape[0]):
        centerd = min(centerd, D - roid_half)
        for centerh in range(roih_half, H + roih_half, label_shape[1]):
            centerh =  min(centerh, H - roih_half)
            for centerw in range(roiw_half, W + roiw_half, label_shape[2]):
                centerw =  min(centerw, W - roiw_half)
                roi_center = [centerd, centerh, centerw, int(C/2)]
                sub_image_centers.append(roi_center)
                sub_image = extract_roi_from_nd_volume(img, roi_center, data_shape, fill = 'random')
                sub_image_patches.append(sub_image)

    # inference with image patches
    total_batch = len(sub_image_patches)
    max_mini_batch = int((total_batch + batch_size -1)/batch_size)
    for mini_batch_idx in range(max_mini_batch):
        batch_end_idx = min((mini_batch_idx+1)*batch_size, total_batch)
        batch_start_idx = batch_end_idx - batch_size
        data_mini_batch = sub_image_patches[batch_start_idx:batch_end_idx]
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        prob_mini_batch = sess.run(proby, feed_dict = {x:data_mini_batch})
        
        for batch_idx in range(batch_start_idx, batch_end_idx):
            roi_center = sub_image_centers[batch_idx]
            roi_center[3] = int(class_num/2)
            prob = set_roi_to_nd_volume(prob, roi_center, prob_mini_batch[batch_idx-batch_start_idx])
    return prob

def convert_label(in_volume, label_convert_source, label_convert_target):
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if(source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume>0] = convert_volume[mask_volume>0]
    return out_volume

class TestAgent:
    def __init__(self, config):
        self.config_data = config['data']
        self.config_net  = config['network']
        self.net_params  = config['network_parameter']
        self.config_test = config['testing']
    
    def construct_network(self):
        # 1, definet network
        if(type(self.config_net['net_type']) is str):
            net_class = NetFactory.create(self.config_net['net_type'])
        else:
            print('customized network is used')
            net_class = self.config_net['net_type']
        self.class_num = self.config_net['class_num']
        self.net = net_class(num_classes = self.class_num,
                        parameters  = self.net_params,
                        w_regularizer = None,
                        b_regularizer = None,
                        name = self.config_net['net_name'])
                        
        # 2, construct temporary network
        batch_size = self.config_test.get('batch_size', 1)
        full_data_shape = [batch_size] + self.config_net['patch_shape_x']
        print('full data shape', full_data_shape)
        x = tf.placeholder(tf.float32, shape = full_data_shape)
        predicty = self.net(x, is_training = False, bn_momentum = 1.0)
        
        # 3, load model
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, self.config_test['model_file'])

    def load_test_data(self):
        def get_data_type(typename):
                return tf.float32 if(typename == 'float') else tf.int32
        x_type = get_data_type(self.config_net['patch_type_x'])
        y_type = get_data_type(self.config_net['patch_type_y'])
        self.config_data['data_type'] = {
                'features': {'x': x_type, 'w': x_type},
                'labels':   {'y': y_type}}
        self.config_data['data_shape'] = {
                'features': {'x': self.config_net['patch_shape_x'],
                             'w': self.config_net['patch_shape_y']},
                'labels':   {'y': self.config_net['patch_shape_y']}}

        data_agent = ImageLoader(self.config_data)
        self.test_dataset = data_agent.get_dataset('test')
#            test_iterator = Iterator.from_structure(test_dataset.output_types,
#                                               test_dataset.output_shapes)
#            self.next_test_batch = test_iterator.get_next()
#        self.test_init_op = test_iterator.make_initializer(test_dataset)

    def test_one_volume(self, img):
        # 1, caulculate shape of tensors
        batch_size = self.config_test.get('batch_size', 1)
        data_shape = self.config_net['patch_shape_x']
        label_shape= self.config_net['patch_shape_y']
        class_num  = self.config_net['class_num']
        margin     = [data_shape[i] - label_shape[i] for i in range(len(data_shape))]
    
        # 2, get test mode.
        #    0, (default) use defined tensor shape and original image shape
        #    1, use defined tensor shape, and resize image in 2D to match tensor shape
        #    2, use original image shape, and resize tensor in 2D to match image shape
        #    3, use original image shape, and resize tensor in 3D to match image shape
        shape_mode = self.config_test.get('shape_mode', 1)
        if(shape_mode == 1):
            [D0, H0, W0, C0] = img.shape
            resized_shape = [D0, data_shape[1], data_shape[2], C0]
            resized_img = resize_ND_volume_to_given_shape(img, resized_shape, order = 3)
        else:
            resized_img = img
        [D, H, W, C] = resized_img.shape
        
        # 3, pad input image to desired size when the network requires the input should be
        #    a multiple of size_factor, e.g. 16
        size_factor = self.config_test.get('size_factor',[1,1,1])
        Dr = int(math.ceil(float(D)/size_factor[0])*size_factor[0])
        Hr = int(math.ceil(float(H)/size_factor[1])*size_factor[1])
        Wr = int(math.ceil(float(W)/size_factor[2])*size_factor[2])
        pad_img = np.random.normal(size = [Dr, Hr, Wr, C])
        pad_img[np.ix_(range(D), range(H), range(W), range(C))] = resized_img
        
        if(shape_mode==3):
            data_shape[0] = Dr
            label_shape[0]= Dr - margin[0]
        data_shape[1] = Hr
        data_shape[2] = Wr
        label_shape[1]= Hr - margin[1]
        label_shape[2]= Wr - margin[2]
        full_data_shape  = [batch_size] + data_shape
        full_label_shape = [batch_size] + label_shape
        full_weight_shape = [i for i in full_data_shape]
        full_weight_shape[-1] = 1
        
        # 4, construct graph
        x = tf.placeholder(tf.float32, shape = full_data_shape)
        predicty = self.net(x, is_training = False, bn_momentum = 1.0)
        print('network input  ', x)
        print('network output ', predicty)
        proby = tf.nn.softmax(predicty)
        
        # 3, load model
#        self.sess = tf.InteractiveSession()
#        self.sess.run(tf.global_variables_initializer())
#        saver = tf.train.Saver()
#        saver.restore(self.sess, self.config_test['model_file'])

        # 5, inference
        outputp = volume_probability_prediction_3d_roi(pad_img, data_shape, label_shape,
                                                       class_num, batch_size, self.sess, x, proby)
        outputy = np.asarray(np.argmax(outputp, axis = 3), np.uint16)
        outputy = outputy[np.ix_(range(D), range(H), range(W))]
        if(shape_mode == 1):
            outputy = resize_ND_volume_to_given_shape(outputy, img.shape[:-1], order = 0)
        return outputy

    def test(self):
        # 1, load data
        self.construct_network()
        self.load_test_data()
#        return
        # 2, test each data
        label_source = self.config_data.get('label_convert_source', None)
        label_target = self.config_data.get('label_convert_target', None)
        if(not(label_source is None) and not(label_source is None)):
            assert(len(label_source) == len(label_target))

        for one_data in self.test_dataset:
            img  = one_data['image']
            name = one_data['name']
            print(img.shape, name)
#            pred = self.test_one_volume(img)
#            if (label_source is not None and label_target is not None):
#                pred = convert_label(pred, label_source, label_target)
#            save_name = '{0:}_{1:}.nii.gz'.format(name, self.config_data['output_postfix'])
#            save_array_as_nifty_volume(pred, self.config_data['save_root']+'/'+save_name)
#
