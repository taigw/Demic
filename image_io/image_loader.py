from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import traceback
import os
import math
import numpy as np
import random
from scipy import ndimage
from Demic.image_io.file_read_write import load_nifty_volume_as_array

def search_file_in_folder_list(folder_list, file_name):
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if(os.path.isfile(full_file_name)):
            file_exist = True
            break
    if(file_exist == False):
        raise ValueError('file not exist: {0:}'.format(file_name))
    return full_file_name

class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

class ImageLoader(object):
    """Wrapper for dataset generation given a read function"""
    def __init__(self, config):
        self.config = config
        # data information
        self.data_type  = config['data_type']
        self.data_shape = config['data_shape']
        self.data_root  = config['data_root']
        self.modality_postfix = config['modality_postfix']
        self.with_ground_truth  = config.get('with_ground_truth', False)
        self.with_weight = config.get('with_weight', False)
        self.label_postfix  = config.get('label_postfix', None)
        self.weight_postfix = config.get('weight_postfix', None)
        self.file_postfix = config['file_postfix']
        self.data_names   = config.get('data_names', None)
        self.data_names_val = config.get('data_names_val', None)
        self.batch_size = config.get('batch_size', 5)
        
        self.replace_background_with_random = config.get('replace_background_with_random', False)
        self.check_image_patch_shape()
    
    def get_dataset(self, mode, shuffle = True):
        """
        mode: train, valid or test
        """
        def data_generator():
            patient_names, full_patient_names = self.get_patient_names(mode)
            for i in range(len(full_patient_names)):
                volume_list = []
                for volume_name in full_patient_names[i]['image_names']:
                    volume = load_nifty_volume_as_array(volume_name)
                    volume_list.append(volume)
                x_array = np.asarray(volume_list, np.float32)
                x_array = np.transpose(x_array, [1, 2, 3, 0]) # [D, H, W, C]
                if(mode == 'test'):
                    yield {'features': {'x': x_array, 'name': patient_names[i]}}
                else:
                    w_array = None
                    weight_name = full_patient_names[i]['weight_name']
                    if(weight_name is not None):
                        weight = load_nifty_volume_as_array(weight_name)
                        w_array = np.asarray([weight])
                        w_array = np.transpose(w_array, [1, 2, 3, 0]) # [D, H, W, C]

                    y_array = None
                    label_name = full_patient_names[i]['label_name']
                    if(label_name is not None):
                        label = load_nifty_volume_as_array(label_name)
                        y_array = np.asarray([label])
                        y_array = np.transpose(y_array, [1, 2, 3, 0]) # [D, H, W, C]
                    
                    x_array, w_array, y_array = self.extract_and_augment_patch(
                        x_array, w_array, y_array)

                    if(w_array is None):
                        w_array = np.ones_like(y_array, np.float32)

                    yield {'features': {'x': x_array, 'w': w_array},
                            'labels': {'y': y_array}}
        if(mode == 'test'):
            return data_generator()
        else:
            dataset = tf.data.Dataset.from_generator(data_generator, self.data_type, self.data_shape)
            if(shuffle):
                dataset = dataset.shuffle(self.batch_size*10)
            dataset = dataset.batch(self.batch_size)
            return dataset

    def check_image_patch_shape(self):
        data_shape   = self.data_shape['features']['x']
        label_shape  = self.data_shape['labels']['y']
        assert(len(data_shape) == 4 and len(label_shape) == 4)
        label_margin = []
        for i in range(3):
            assert(data_shape[i] >= label_shape[i])
            margin = (data_shape[i] - label_shape[i]) % 2
            assert( margin == 0)
            label_margin.append(margin)
        label_margin.append(0)
        self.label_margin = label_margin

    def get_patient_names(self, mode):
        """
            mode: train, valid, or test
        """
        if(mode == 'train' or mode == 'test'):
            data_names = self.data_names
        elif(mode =='valid'):
            data_names = self.data_names_val
        else:
            raise ValueError("mode should be train, valid, or test, {0:} received".format(mode))
        assert(os.path.isfile(data_names))
        with open(data_names) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
        
        full_patient_names = []
        for i in range(len(patient_names)):
            image_names = []
            for mod_idx in range(len(self.modality_postfix)):
                image_name_short = patient_names[i] + '_' + \
                                self.modality_postfix[mod_idx] + '.' + \
                                self.file_postfix
                image_name = search_file_in_folder_list(self.data_root, image_name_short)
                image_names.append(image_name)
            weight_name = None
            if(self.with_weight):
                weight_name_short = patient_names[i] + '_' + \
                                    self.weight_postfix + '.' + \
                                    self.file_postfix
                weight_name = search_file_in_folder_list(self.data_root, weight_name_short)
            label_name = None
            if(self.with_ground_truth):
                label_name_short = patient_names[i] + '_' + \
                                    self.label_postfix + '.' + \
                                    self.file_postfix
                label_name = search_file_in_folder_list(self.data_root, label_name_short)
            one_patient_names = {}
            one_patient_names['image_names'] = image_names
            one_patient_names['weight_name'] = weight_name
            one_patient_names['label_name'] = label_name
            full_patient_names.append(one_patient_names)
        return patient_names, full_patient_names

    def pad_volume_to_desired_shape(self, volume, desired_shape):
        """ Pad a volume to desired shape
            if the input size is larger than output shape, then reture then input volume
        """
        input_shape = np.shape(volume)
        output_shape = [max(input_shape[i], desired_shape[i]) for i in range(len(input_shape))]
        pad = [math.ceil((output_shape[i]-input_shape[i])/2) for i in range(len(input_shape))]
        if(np.asarray(pad).sum() > 0):
            volume = np.pad(volume, pad, mode = 'reflect')
        return volume
    
    def random_sample_patch(self, img, wht, lab):
        """Sample a patch from the image with a random position.
            The output size of img_sub and label_sub may not be the same.
            image, label are sampled with the same central voxel.
        """
        img_shape_out = self.config['data_shape']['features']['x']
        lab_shape_out = self.config['data_shape']['labels']['y']
        
        # if output shape is larger than input shape, padding is needed
        img = self.pad_volume_to_desired_shape(img, img_shape_out)
        lab = self.pad_volume_to_desired_shape(lab, lab_shape_out)
        img_shape_in = np.shape(img)
        lab_shape_in = np.shape(lab)

        img_begin = []
        lab_begin = []
        for i in range(len(img_shape_in)):
            img_begin_i = int(random.random()*(img_shape_in[i] - img_shape_out[i]))
            lab_begin_i = img_begin_i + self.label_margin[i]
            img_begin.append(img_begin_i)
            lab_begin.append(lab_begin_i)

        img_sub = img[img_begin[0]:img_begin[0] + img_shape_out[0],
                      img_begin[1]:img_begin[1] + img_shape_out[1],
                      img_begin[2]:img_begin[2] + img_shape_out[2],:]
        lab_sub = lab[lab_begin[0]:lab_begin[0] + lab_shape_out[0],
                      lab_begin[1]:lab_begin[1] + lab_shape_out[1],
                      lab_begin[2]:lab_begin[2] + lab_shape_out[2],:]
        wht_sub = None
        if(wht is not None):
            wht_sub = wht[lab_begin[0]:lab_begin[0] + lab_shape_out[0],
                          lab_begin[1]:lab_begin[1] + lab_shape_out[1],
                          lab_begin[2]:lab_begin[2] + lab_shape_out[2],:]
        return [img_sub, wht_sub, lab_sub]
    
    def get_4d_bounding_box(self, label, margin):
        [d_idxes, h_idxes, w_idxes, w_idxes] = np.nonzero(label)
        mind = d_idxes.min(); maxd = d_idxes.max()
        minh = h_idxes.min(); maxh = h_idxes.max()
        minw = w_idxes.min(); maxw = w_idxes.max()
        minc = c_idxes.min(); maxc = c_idxes.max()
        
        idx_min = [mind, minh, minw, minc]
        idx_max = [maxd, maxh, maxw, maxc]
        input_shape = np.shape(label)
        for i in range(4):
            idx_min[i] = max(idx_min[i] - margin[i], 0)
            idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
        return idx_min, idx_max

    def crop_4d_volume_with_bounding_box(self, volume, min_idx, max_idx):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
        return output

    def extract_and_augment_patch(self, x_array, w_array, y_array):
        """
            Extract patches from loaded image and label volumes
            Args:
                x_array: an image volume, numpy array with shape [D, H, W, C]
                w_array: an image volume, numpy array with shape [D, H, W, C]
                y_array: a label volume, numpy array with shape [D, H, W, C]
            Returns:
                x_array: a subregion of input x_array
                y_array: a subregion of input y_array
            
        """
        # 1, augmentation by random rotation
        random_rotate = self.config.get('random_rotate', None)
        if(not(random_rotate is None)):
            assert(len(random_rotate) == 2 and (random_rotate[0] < random_rotate[1]))
            angle = random.uniform(random_rotate[0], random_rotate[1])
            x_array = ndimage.interpolation.rotate(x_array, angle, (1,2), order = 2)
            if(w_array is not None):
                w_array = ndimage.interpolation.rotate(w_array, angle, (1,2), order = 2)
            if(y_array is not None):
                y_array = ndimage.interpolation.rotate(y_array, angle, (1,2), order = 0)

        # 2, extract image patch
        patch_mode = self.config.get('patch_mode', 0)
        if(patch_mode == 0): # randomly sample patch with fixed size
            [x_array, w_array, y_array] = self.random_sample_patch(x_array, w_array, y_array)
        elif(patch_mode == 1):
            # crop with 3d bounding box, resize to given 2D size, then sample along z-axis
            assert(y_array is not None)
            margin = self.config.get('bounding_box_margin', [0,0,0])
            [min_idx, max_idx] = self.get_4d_bounding_box(y_array, margin + [0])
            x_array = self.crop_4d_volume_with_bounding_box(x_array, min_idx, max_idx)
            if(w_array is not None):
                w_array = self.crop_4d_volume_with_bounding_box(w_array, min_idx, max_idx)
            y_array = self.crop_4d_volume_with_bounding_box(y_array, min_idx, max_idx)
        else:
            [x_array, w_array, y_array] = self.random_sample_patch(x_array, w_array, y_array)

        # 3, augment by random flip
        if(self.config.get('flip_w', False) and (random.random() > 0.5)):
            x_array = x_array[:, :, ::-1, :]
            if(w_array is not None):
                w_array = w_array[:, :, ::-1, :]
            if(y_array is not None):
                y_array = y_array[:, :, ::-1, :]
        if(self.config.get('flip_h', False) and (random.random() > 0.5)):
            x_array = x_array[:, ::-1, :, :]
            if(w_array is not None):
                w_array = w_array[:, ::-1, :, :]
            if(y_array is not None):
                y_array = y_array[:, ::-1, :, :]

        # 4, convert label
        label_source = self.config.get('label_convert_source', None)
        label_target = self.config.get('label_convert_target', None)
        if((label_source is not None) and (label_target is not None)):
            assert(y_array is not None)
            assert(len(label_source) == len(label_target))
            label_converted = np.zeros_like(y_array)
            for i in range(len(label_source)):
                label_temp = y_array == np.ones_like(y_array)*label_source[i]
                label_temp = np.asarray(label_temp, np.int32)*label_target[i]
                label_converted = label_converted + label_temp
            y_array = label_converted
        return x_array, w_array, y_array
    

    def get_inputs(self, mode):
        """
        Function to provide the input_fn for a tf.Estimator.

        Args:
            mode: A tf.estimator.ModeKeys.
        Returns:
            function: a handle to the `input_fn` to be passed the relevant
                tf estimator functions.
            tf.train.SessionRunHook: A hook to initialize the queue within
                the dataset.
        """
        def data_generator():
            patient_names, full_patient_names = self.get_patient_names(mode)
            for i in range(len(full_patient_names)):
                volume_list = []
                for volume_name in full_patient_names[i]['image_names']:
                    volume = load_nifty_volume_as_array(volume_name)
                    volume_list.append(volume)
                x_array = np.asarray(volume_list, np.float32)
                x_array = np.transpose(x_array, [1, 2, 3, 0]) # [D, H, W, C]
                
                label_name = full_patient_names[i]['label_name']
                label = load_nifty_volume_as_array(label_name)
                y_array = np.asarray([label])
                y_array = np.transpose(y_array, [1, 2, 3, 0]) # [D, H, W, C]
                
                x_array, y_array = self.extract_and_augment_patch(x_array, y_array)
                yield {'features': {'x': x_array},
                        'labels': {'y': y_array}}

        def train_inputs():
            dataset = tf.data.Dataset.from_generator(
                data_generator, self.data_type, self.data_shape)
            dataset = dataset.repeat(None)
            dataset = dataset.shuffle(self.batch_size*20)
            dataset = dataset.batch(self.batch_size)

            iterator = dataset.make_initializable_iterator()
            next_dict = iterator.get_next()

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(iterator.initializer)

            # Return batched (features, labels)
            return next_dict['features'], next_dict.get('labels')

        # Return function and hook
        iterator_initializer_hook = IteratorInitializerHook()
        return train_inputs, iterator_initializer_hook
    
    def serving_input_receiver_fn(self, placeholder_shapes):
        """Build the serving inputs.

        Args:
            placeholder_shapes: A nested structure of lists or tuples
                corresponding to the shape of each component of the feature
                elements yieled by the read_fn.

        Returns:
            function: A function to be passed to the tf.estimator.Estimator
            instance when exporting a saved model with estimator.export_savedmodel.
        """

        def f():
            inputs = {k: tf.placeholder(
                shape=[None] + list(placeholder_shapes['features'][k]),
                dtype=self.data_type['features'][k])
                      for k in list(self.data_type['features'].keys())}

            return tf.estimator.export.ServingInputReceiver(inputs, inputs)
        return f

if __name__=="__main__":
    gen = data_generator()
    for i in range(10):
        d = gen.next()
        print(d['features']['x'].shape)
