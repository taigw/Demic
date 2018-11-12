# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys
from Demic.net.unet2d import UNet2D
from Demic.net.unet2d5 import UNet2D5
from Demic.net.fcn2d import FCN2D
from Demic.net.pnet import PNet
from Demic.net.pnet_stn_fuse import PNet_STN_DF
from Demic.net.vgg21 import VGG21
class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'UNet2D':
            return UNet2D
        if name == 'UNet2D5':
            return UNet2D5
        if name == 'FCN2D':
            return FCN2D
        if name == 'PNet':
            return PNet
        if name == 'PNet_STN_DF':
            return PNet_STN_DF
        if name == 'VGG21':
            return VGG21
        print('unsupported network:', name)
        exit()
