# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys
from Demic.net.unet2d import UNet2D
from Demic.net.pnet import PNet
from Demic.net.pnet_multi_slice import PNet_Multi_Slice
from Demic.net.pnet_multi_slice_deep_fuse import PNet_Multi_Slice_Deep_Fuse
from Demic.net.pnet_stn import PNet_STN
from Demic.net.pnet_stn_deep_fuse import PNet_STN_DF
from Demic.net.pnet_stn_deep_weight_fuse import PNet_STN_WDF
from Demic.net.vgg21 import VGG21
class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'UNet2D':
            return UNet2D
        if name == 'PNet':
            return PNet
        if name == 'PNet_Multi_Slice':
            return PNet_Multi_Slice
        if name == 'PNet_Multi_Slice_Deep_Fuse':
            return PNet_Multi_Slice_Deep_Fuse
        if name == 'PNet_STN':
            return PNet_STN
        if name == 'PNet_STN_DF':
            return PNet_STN_DF
        if name == 'PNet_STN_WDF':
            return PNet_STN_WDF
        if name == 'VGG21':
            return VGG21
        print('unsupported network:', name)
        exit()
