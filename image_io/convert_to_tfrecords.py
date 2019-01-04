# Created on Wed Oct 11 2017
#
# @author: Guotai Wang
# reference: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

import os
import sys
from scipy import ndimage
import numpy as np
import nibabel
import tensorflow as tf
from PIL import Image

from Demic.util.parse_config import parse_config
from Demic.image_io.file_read_write import *
from Demic.util.image_process import *


class DataLoader():
    def __init__(self, config):
        self.config = config
        # data information
        self.data_root = config['data_root']
        self.modality_postfix = config.get('modality_postfix', None)
        self.with_ground_truth  = config.get('with_ground_truth', False)
        self.with_weight = config.get('with_weight', False)
        self.image_file_postfix  = config['image_file_postfix']
        self.label_file_postfix  = config.get('label_file_postfix', self.image_file_postfix)
        self.weight_file_postfix = config.get('weight_file_postfix', self.image_file_postfix)
        self.label_postfix  = config.get('label_postfix', None)
        self.weight_postfix = config.get('weight_postfix', None)
        
        self.data_names = config.get('data_names', None)
        self.data_subset = config.get('data_subset', None)
        self.replace_background_with_random = config.get('replace_background_with_random', False)

    def __get_patient_names(self):
        if(not(self.data_names is None)):
            assert(os.path.isfile(self.data_names))
            with open(self.data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content] 
        else: # load all image in data_root
            sub_dirs = [x[0] for x in os.walk(self.data_root[0])]
            print(sub_dirs)
            patient_names = []
            for sub_dir in sub_dirs:
                names = os.listdir(sub_dir)
                if(sub_dir == self.data_root[0]):
                    sub_patient_names = []
                    for x in names:
                        if(self.image_file_postfix in x):
                            idx = x.rfind('_')
                            xsplit = x[:idx]
                            sub_patient_names.append(xsplit)
                else:
                    sub_dir_name = sub_dir[len(self.data_root[0])+1:]
                    sub_patient_names = []
                    for x in names:
                        if(self.image_file_postfix in x):
                            idx = x.rfind('_')
                            xsplit = os.path.join(sub_dir_name,x[:idx])
                            sub_patient_names.append(xsplit)                    
                sub_patient_names = list(set(sub_patient_names))
                sub_patient_names.sort()
                patient_names.extend(sub_patient_names)   
        return patient_names    

    def load_data(self):
        patient_names = self.__get_patient_names()
        if(not(self.data_subset is None)):
            patient_names = patient_names[self.data_subset[0]:self.data_subset[1]]
        self.patient_names = patient_names
        file_names = []
        X = []
        W = []
        Y = []
        image_info = []
        for i in range(len(self.patient_names)):
            print(i, self.patient_names[i])
            if(self.with_weight):
                weight_name_short = self.patient_names[i] + '_' + self.weight_postfix + \
                                     '.' + self.weight_file_postfix
                weight_name = search_file_in_folder_list(self.data_root, weight_name_short)
                w_array = load_image_as_4d_array(weight_name)['data_array']
                w_array = np.asarray(w_array, np.float32)
                W.append(w_array)      
            if(self.with_ground_truth):
                label_name_short = self.patient_names[i] + '_' + self.label_postfix + \
                                     '.' + self.label_file_postfix
                label_name = search_file_in_folder_list(self.data_root, label_name_short)
                y_array  = load_image_as_4d_array(label_name)['data_array']
                Y.append(y_array)  
            volume_list = []
            file_list   = []
            if (self.modality_postfix is None): # single modality
                volume_name_short = self.patient_names[i] +  '.' + self.image_file_postfix
                volume_name = search_file_in_folder_list(self.data_root, volume_name_short)
                volume_dict  = load_image_as_4d_array(volume_name)
                volume_array = np.asarray(volume_dict['data_array'],  np.float32)
                temp_img_info = {'spacing': volume_dict['spacing'],
                                 'direction':volume_dict['direction']}
                if(self.with_weight and self.replace_background_with_random):
                        arr_random = np.random.normal(0, 1, size = volume_array.shape)
                        volume_array[w_array==0] = arr_random[volume_array==0]
                file_list.append(volume_name)
            else: # multiple modality
                for mod_idx in range(len(self.modality_postfix)):
                    volume_name_short = self.patient_names[i] + '_' + self.modality_postfix[mod_idx] + \
                                        '.' + self.image_file_postfix
                    volume_name = search_file_in_folder_list(self.data_root, volume_name_short)
                    volume_dict = load_image_as_4d_array(volume_name)
                    volume_array_i = np.asarray(volume_dict['data_array'], np.float32)
                    if(mod_idx == 0):
                        temp_img_info = {'spacing': volume_dict['spacing'],
                                 'direction':volume_dict['direction']}
                    if(self.with_weight and self.replace_background_with_random):
                        arr_random = np.random.normal(0, 1, size = volume_array_i.shape)
                        volume_array_i[w_array==0] = arr_random[w_array==0]
                    volume_list.append(volume_array_i)
                    file_list.append(volume_name)
                volume_array = np.concatenate(volume_list, axis = -1)
            # for intensity normalize
            intensity_normalize_mode = self.config.get('intensity_normalize_mode', 0)
            if (intensity_normalize_mode == 0):
                pass
            elif(intensity_normalize_mode == 1): # use given mean and std
                iten_mean = np.asarray(self.config['intensity_normalize_mean'])
                iten_std  = np.asarray(self.config['intensity_normalize_std'])
                assert(iten_mean.size == volume_array.shape[-1])
                assert(iten_std.size == volume_array.shape[-1])
                volume_array = (volume_array - iten_mean)/iten_std
            elif(intensity_normalize_mode == 2): # use mean and std based on mask
                print("intensity normalize mode 2")
                mask = None 
                if(self.config.get('use_mask', False)):
                    use_nonzero_weight_as_mask = self.config.get('use_nonzero_weight_as_mask', False)
                    if(use_nonzero_weight_as_mask):
                        mask = w_array > 0
                    else:
                        mask = volume_array[:, :, :, 0] > self.config['intensity_threshold_for_mask']
                for c in range(volume_array.shape[-1]):
                    volume_array[:, :, :, c] = itensity_normalize_one_volume( \
                                                 volume_array[:, :, :, c] ,  mask, True)
            else:
                raise ValueError("Not implemented: intensity_normalize_mode " + \
                                    "{0:}".format(intensity_normalize_mode))
            X.append(volume_array)
            file_names.append(file_list)
            image_info.append(temp_img_info)
        print('{0:} volumes have been loaded'.format(len(self.patient_names)))
        self.data   = X
        self.weight = W
        self.label  = Y
        self.file_names = file_names
        self.image_info = image_info
    
    def get_image_number(self):
        return len(self.patient_names)

    def get_image(self, idx, with_ground_truth = True):
        if(with_ground_truth and self.with_ground_truth):
            label = self.label[idx]
        else:
            label = None
        if(self.with_weight):
            weight = self.weight[idx]
        else:
            weight = None
        output = [self.patient_names[idx], self.file_names[idx],
                  self.data[idx], weight, label, self.image_info[idx]]
        return output

    def save_to_tfrecords(self):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        tfrecords_filename = self.config['tfrecords_filename']
        tfrecord_options= tf.python_io.TFRecordOptions(1)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename, tfrecord_options)
        for i in range(len(self.data)):
            feature_dict = {}
            img    = np.asarray(self.data[i], np.float32)
            img_raw    = img.tostring()
            img_shape    = np.asarray(img.shape, np.int32)
            img_shape_raw    = img_shape.tostring()
            feature_dict['image_raw'] = _bytes_feature(img_raw)
            feature_dict['image_shape_raw'] = _bytes_feature(img_shape_raw)
            assert(len(img_shape) == 4)
            if(self.with_weight):
                weight = np.asarray(self.weight[i], np.float32)
                weight_raw = weight.tostring()
                weight_shape = np.asarray(weight.shape, np.int32)
                weight_shape_raw = weight_shape.tostring()
                feature_dict['weight_raw'] = _bytes_feature(weight_raw)
                feature_dict['weight_shape_raw'] = _bytes_feature(weight_shape_raw)
                assert(len(weight_shape) == 4)
                assert(img_shape[0] == weight_shape[0])
                assert(img_shape[1] == weight_shape[1])
                assert(img_shape[2] == weight_shape[2])
            if(self.with_ground_truth):
                label  = np.asarray(self.label[i], np.int32)
                label_raw  = label.tostring()
                label_shape  = np.asarray(label.shape, np.int32)
                label_shape_raw  = label_shape.tostring()
                feature_dict['label_raw'] = _bytes_feature(label_raw)
                feature_dict['label_shape_raw'] = _bytes_feature(label_shape_raw)
                assert(len(label_shape) == 4)
                assert(img_shape[0] == label_shape[0])
                assert(img_shape[1] == label_shape[1])
                assert(img_shape[2] == label_shape[2])
            example = tf.train.Example(features=tf.train.Features(feature = feature_dict))
            writer.write(example.SerializeToString())
        writer.close()

def convert_to_rf_records(config_file):
    config = parse_config(config_file)
    config_data = config['data']
    data_loader = DataLoader(config_data)
    data_loader.load_data()
    data_loader.save_to_tfrecords()
if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('    python convert_to_tfrecords.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    convert_to_rf_records(config_file)

