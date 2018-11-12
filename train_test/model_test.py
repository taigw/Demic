"""Script for testing
Author: Guotai Wang
"""

import os
import sys
import math
import time
import numpy as np
from PIL import Image
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from Demic.net.net_factory import NetFactory
from Demic.image_io.file_read_write import *
from Demic.image_io.convert_to_tfrecords import DataLoader
from Demic.train_test.test_func import volume_probability_prediction_3d_roi
from Demic.util.parse_config import parse_config
from Demic.util.image_process import *

class TestAgent:
    def __init__(self, config):
        self.config_data = config['data']
        self.config_net  = config['network']
        self.config_test = config['testing']
        
        # creat net
        net_class = NetFactory.create(self.config_net['net_type'])
        self.net  = net_class(num_classes = self.config_net['class_num'],
                             parameters = config['network_parameter'],
                             w_regularizer = None,
                             b_regularizer = None,
                             name = self.config_net['net_name'])

    def construct_network(self):
        shape_mode     = self.config_test.get('shape_mode', 1)
        fix_batch_size = self.config_test.get('fix_batch_size', True)
        batch_size     = self.config_test.get('batch_size', 1)
        data_shape = self.config_net['data_shape']
        label_shape= self.config_net['label_shape']
        class_num  = self.config_net['class_num']
        bn_training_mode = self.config_net.get('bn_training_mode', False)
        
        margin = [data_shape[i] - label_shape[i] for i in range(len(data_shape))]
        full_data_shape  = ([batch_size] if fix_batch_size else [None]) + data_shape
        
        self.x = tf.placeholder(tf.float32, shape = full_data_shape)
        print('network input', self.x)
        predicty = self.net(self.x, is_training = bn_training_mode, bn_momentum = 0.0)
        print('network output shape ', predicty)
        self.proby = tf.nn.softmax(predicty)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        all_vars = tf.global_variables()
        restore_vars= [vars for vars in all_vars if self.config_net['net_name'] in vars.name]
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, self.config_net['model_file'])

    def test_one_volume(self, img):
        # calculate shape of tensors
        batch_size = self.config_test.get('batch_size', 1)
        data_shape = self.config_net['data_shape']
        label_shape= self.config_net['label_shape']
        class_num  = self.config_net['class_num']
        resize_input = self.config_test.get('resize_input', False)
        resize_input_to_integer_factor = self.config_test.get('resize_input_to_integer', None)
        resize_input_to_given_shape_in_2d = self.config_test.get('resize_input_to_given_shape_in_2d', None)
        resize_input_to_given_shape_in_3d = self.config_test.get('resize_input_to_given_shape_in_3d', None)
        use_depth_as_batch_size = self.config_test.get('use_depth_as_batch_size', False)

        if(resize_input):
            [D0, H0, W0, C0] = img.shape
            if(resize_input_to_integer_factor is not None):
                assert(len(resize_input_to_integer_factor) == 3)
                resize_factor = resize_input_to_integer_factor
                Dr = int(math.ceil(float(D0)/resize_factor[0])*resize_factor[0])
                Hr = int(math.ceil(float(H0)/resize_factor[1])*resize_factor[1])
                Wr = int(math.ceil(float(W0)/resize_factor[2])*resize_factor[2])
            elif(resize_input_to_given_shape_in_2d is not None):
                assert(len(resize_input_to_given_shape_in_2d) == 2)
                Dr = D0
                [Hr, Wr] = resize_input_to_given_shape_in_2d
            elif(resize_input_to_given_shape_in_3d is not None):
                assert(len(resize_input_to_given_shape_in_3d) == 3)
                [Dr, Hr, Wr] = resize_input_to_given_shape_in_3d
            else:
                raise ValueError("parameters for input resize should be provided")
            resized_img_shape = [Dr, Hr, Wr, C0]
            resized_img = resize_ND_volume_to_given_shape(img, resized_img_shape, order = 3)
        else:
            resized_img = img
        
        if (use_depth_as_batch_size):
            batch_size = min(resized_img.shape[0], self.config_test['batch_size'])
        
        # inference
        outputp = volume_probability_prediction_3d_roi(resized_img, data_shape, label_shape,
                                                class_num, batch_size, self.sess, self.x, self.proby)
        outputp = outputp[:, :, :, -1] # get foreground probability
        if(resize_input):
            outputp = resize_ND_volume_to_given_shape(outputp, list(img.shape[:-1]), order = 1)
        return outputp

    def test(self):
        random.seed(100)
        self.construct_network()
        data_loader = DataLoader(self.config_data)
        data_loader.load_data()

        label_source = self.config_data.get('label_convert_source', None)
        label_target = self.config_data.get('label_convert_target', None)

        if(not(label_source is None) and not(label_source is None)):
            assert(len(label_source) == len(label_target))
        img_num = data_loader.get_image_number()
        test_time = []
        print('image number', img_num)
        for i in range(img_num):
            [patient_name, file_names, img_raw, weight_raw, lab_raw, spacing] = data_loader.get_image(i)
            print(patient_name, lab_raw.shape)
            # iten_mean = [179.69427237, 146.44891944, 134.39686832]
            # iten_std  = [40.37515566, 42.92464467, 46.74197245]
            # img = img_raw[0] * np.asarray(iten_std) + iten_mean
            # img = Image.fromarray(np.asarray(img, np.uint8), 'RGB')
            # img.save("result/unet/{0:}_seg.png".format(patient_name))
            t0  = time.time()
            outp = self.test_one_volume(img_raw)
            out = np.asarray(outp >= 0.5, np.uint8)

            test_time.append(time.time() - t0)
            if(not(label_source is None) and not(label_source is None)):
                out = convert_label(out, label_source, label_target)

            save_name = "{0:}_{1:}.{2:}".format(patient_name, 
                                                 self.config_data['output_postfix'],
                                                 self.config_data['outputfile_postfix'])
            if (self.config_data['outputfile_postfix'] == "nii.gz" or self.config_data['outputfile_postfix'] == "nii"):
                save_array_as_nifty_volume(out, self.config_data['save_root']+'/'+save_name, file_names[0])
            elif(self.config_data['outputfile_postfix'] == "jpg" or self.config_data['outputfile_postfix'] == "png"):
                assert(out.shape[0] == 1 and len(out.shape) == 3)
                out = np.reshape(out, out.shape[1:])
                if(self.config_net['class_num'] == 2):
                    out = out * 255
                out_img = Image.fromarray(out, 'L')
                out_img.save(self.config_data['save_root'] + '/' + save_name)
                print("save array with shape", out.shape)
            if(self.config_test.get('save_probability', False)):
                save_name = '{0:}_{1:}.nii.gz'.format(patient_name, 'Prob')
                save_array_as_nifty_volume(outp, self.config_data['save_root']+'/'+save_name, file_names[0])
        test_time = np.asarray(test_time)
        print('test time', test_time.mean(), test_time.std())
        np.savetxt("{0:}/test_time.txt".format(self.config_data['save_root']), test_time)

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train_test/model_test.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    config = parse_config(config_file)
    test_agent = TestAgent(config)
    test_agent.test()
