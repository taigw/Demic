# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys
from net.unet2d import UNet2D
from net.pnet import PNet
class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'UNet2D':
            return UNet2D
        if name == 'PNet':
            return PNet
        print('unsupported network:', name)
        exit()
