"""Script for testing
Author: Guotai Wang
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util.parse_config import parse_config
from datetime import datetime
from net.net_factory import NetFactory
from image_io.file_read_write import *
from image_io.convert_to_tfrecords import itensity_normalize_one_volume

def test_volume(img, config):
    [D, H, W] = img.shape
    print('input image shape', img.shape)
    config_net  = config['network']
    config_test = config['testing']
    
    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    batch_size  = 1 #config_tfrecords.get('batch_size', 1)
    #    full_data_shape  = [batch_size] + config_net['data_shape']
    #    full_label_shape = [batch_size] + config_net['label_shape']
    full_data_shape  = [batch_size] + [1, H, W, 1]
    full_label_shape = [batch_size] + [1, H, W, 1]
    data_channel= full_data_shape[-1]
    class_num   = config_net['class_num']
    
    full_weight_shape = [i for i in full_data_shape]
    full_weight_shape[-1] = 1
    data_space_shape  = full_data_shape[:-1]
    label_space_shape = full_label_shape[:-1]
    
    # construct graph
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_weight_shape)
    y = tf.placeholder(tf.int32, shape = full_label_shape)
    
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
                    w_regularizer = None,
                    b_regularizer = None,
                    name = net_name)
    predicty = net(x, is_training = True)
    print('network output shape ', predicty)
    proby = tf.nn.softmax(predicty)
    
                    
    # start the session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, config_net['model_file'])
    outputy = []
    for i in range(D):
        print(i)
        img_slice = img[i]
        slice = np.reshape(img_slice, [1, 1, H, W, 1])
        proby = sess.run(predicty, feed_dict = {x:slice})[0]
        predy =  np.asarray(np.argmax(proby, axis = 3), np.uint16)[0]
        outputy.append(predy)
    outputy = np.asarray(outputy, np.uint8)
    return outputy

def model_test(config_file):
    img = load_nifty_volume_as_array('/Users/guotaiwang/Documents/data/FetalBrain/Valid/17_10_10_Image.nii.gz')
    img = np.asarray(img, np.float32)
    img = itensity_normalize_one_volume(img, 0)
    config = parse_config(config_file)
    out = test_volume(img, config)
    save_array_as_nifty_volume(out, '../result/17_10_10_Seg1.nii.gz')

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python model_test.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    model_test(config_file)
