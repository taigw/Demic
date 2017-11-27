# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from scipy import ndimage

def itensity_normalize_one_volume(volume, mask = None, replace = False):
    """
        normalize a volume image with mean dand std of the mask region
        """
    if(mask is None):
        mask = volume > 0
    pixels = volume[mask>0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    if(replace):
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[mask==0] = out_random[mask==0]
    return out

def resize_ND_volume_to_given_shape(volume, out_shape, order = 3):
    shape0=volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    return ndimage.interpolation.zoom(volume, scale, order = order)

def get_ND_bounding_box(label, margin):
    """
    get the bounding box of an ND binary volume
    """
    input_shape = label.shape
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

def pad_ND_volume_to_desired_shape(volume, desired_shape, mode = 'reflect'):
    """ Pad a volume to desired shape
        if the input size is larger than output shape, then reture then input volume
    """
    input_shape = volume.shape
    output_shape = [max(input_shape[i], desired_shape[i]) for i in range(len(input_shape))]
    pad_width = []
    pad_flag  = False
    for i in range(len(input_shape)):
        pad_lr = output_shape[i]-input_shape[i]
        if(pad_lr > 0):
            pad_flag = True
        pad_l  = int(pad_lr/2)
        pad_r  = pad_lr - pad_l
        pad_width.append((pad_l, pad_r))
    if(pad_flag):
        volume = np.pad(volume, pad_width, mode = mode)
    return volume

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output
