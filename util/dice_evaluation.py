# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from scipy import ndimage
from Demic.util.image_process import *
from Demic.util.parse_config import parse_config
from Demic.image_io.file_read_write import *


def binary_dice(s, g, resize = False):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    resize: if s and g have different shapes, resize s to match g.
    """
    assert(len(s.shape)== len(g.shape))
    if(resize):
        size_match = True
        for i in range(len(s.shape)):
            if(s.shape[i] != g.shape[i]):
                size_match = False
                break
        if(size_match is False):
            s = resize_ND_volume_to_given_shape(s, g.shape, order = 0)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = 2.0*s0/(s1 + s2 + 0.00001)
    return dice

def dice_of_binary_volumes(s_name, g_name):
    s = load_nifty_volume_as_array(s_name)
    g = load_nifty_volume_as_array(g_name)
    dice = binary_dice3d(s, g)
    return dice

def dice_evaluation(config_file):
    config = parse_config(config_file)['evaluation']
    labels = config['label_list']
    label_convert_source = config.get('label_convert_source', None)
    label_convert_target = config.get('label_convert_target', None)
    s_folder = config['segmentation_folder']
    g_folder = config['ground_truth_folder']
    s_postfix = config.get('segmentation_postfix',None)
    g_postfix = config.get('ground_truth_postfix',None)
    s_format  = config.get('segmentation_format', "nii.gz")
    g_format  = config.get('ground_truth_format', "nii.gz")
    remove_outlier = config.get('remove_outlier', False)
    file_postfix   = config.get('file_postfix','dice')

    s_postfix_long = '.' + s_format
    if(s_postfix is not None):
        s_postfix_long = '_' + s_postfix + s_postfix_long
    g_postfix_long = '.' + g_format
    if(g_postfix is not None):
        g_postfix_long = '_' + g_postfix + g_postfix_long

    patient_names_file = config['patient_file_names']
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content] 
    dice_all_data = []
    for i in range(len(patient_names)):
        s_name = os.path.join(s_folder, patient_names[i] + s_postfix_long)
        g_name = os.path.join(g_folder, patient_names[i] + g_postfix_long)
        s_volume, _ = load_image_as_array(s_name)
        g_volume, _ = load_image_as_array(g_name)
        if((label_convert_source is not None) and label_convert_target is not None):
            s_volume = convert_label(s_volume, label_convert_source, label_convert_target)

        # fuse multiple labels
        s_volume_sub = np.zeros_like(s_volume)
        g_volume_sub = np.zeros_like(g_volume)
        for lab in labels:
            s_volume_sub = s_volume_sub + np.asarray(s_volume == lab, np.uint8)
            g_volume_sub = g_volume_sub + np.asarray(g_volume == lab, np.uint8)
#        if(s_volume_sub.sum() > 0):
#            s_volume_sub = get_largest_component(s_volume_sub)
        if(remove_outlier):
            strt = ndimage.generate_binary_structure(3,2) # iterate structure
            post = ndimage.morphology.binary_closing(s_volume_sub, strt)
            post = get_largest_component(post)
            s_volume_sub = np.asarray(post*s_volume_sub, np.uint8)
        temp_dice = binary_dice(s_volume_sub > 0, g_volume_sub > 0)
        dice_all_data.append(temp_dice)
        print(patient_names[i], temp_dice)
    dice_all_data = np.asarray(dice_all_data)
    dice_mean = [dice_all_data.mean(axis = 0)]
    dice_std  = [dice_all_data.std(axis = 0)]
    np.savetxt("{0:}/{1:}_all.txt".format(s_folder, file_postfix), dice_all_data)
    np.savetxt("{0:}/{1:}_mean.txt".format(s_folder, file_postfix), dice_mean)
    np.savetxt("{0:}/{1:}_std.txt".format(s_folder, file_postfix), dice_std)
    print('dice mean ', dice_mean)
    print('dice std  ', dice_std)
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python util/dice_evaluation.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    dice_evaluation(config_file)
