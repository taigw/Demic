# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from scipy import ndimage


## for 2D images
def resize_2d_image_to_fixed_width(img, width, order = 1):
    [H, W] = img.shape
    scale  = float(width)/W
    resized_img = ndimage.interpolation.zoom(img, scale, order = order)
    return resized_img

def resize_2d_image_to_fixed_height(img, height, order = 1):
    [H, W] = img.shape
    scale  = float(height)/H
    resized_img = ndimage.interpolation.zoom(img, scale, order = order)
    return resized_img

def scale_itensity_for_visualize(img):
    minv = img.min()
    maxv = img.max()
    output = (img - minv)*255/(maxv-minv)
    output = np.uint8(output)
    return output

## for 3D images

## for ND images
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

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif(dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif(dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out

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
