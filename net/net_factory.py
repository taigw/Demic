# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys
from net.unet2d import UNet2D

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'UNet2D':
            return UNet2D
        print('unsupported network:', name)
        exit()
