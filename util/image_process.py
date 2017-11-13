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

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python util/dice_evaluation.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    dice_evaluation(config_file)
