# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys
from net.unet2d import UNet2D
from net.pnet import PNet
from net.pnet_ms import PNet_Multi_Slice
from net.vgg21 import VGG21
class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'UNet2D':
            return UNet2D
        if name == 'PNet':
            return PNet
        if name == 'PNet_Multi_Slice':
            return PNet_Multi_Slice
        if name == 'VGG21':
            return VGG21
        print('unsupported network:', name)
        exit()
