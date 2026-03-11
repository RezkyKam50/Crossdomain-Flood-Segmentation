import math
import torch
import torch.nn as nn

from copy import deepcopy
from torch import nn
import torch
import numpy as np
from typing import Tuple, List
from torch.cuda.amp import autocast
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

from models.blocks import DownBlock, UpBlockWithSkip, Block



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, unet_encoder_size=None):
        super(UNet, self).__init__()

        if unet_encoder_size is None:
            unet_encoder_size = in_channels * 128

        # Encoder
        self.down1 = DownBlock(in_channels, in_channels*4) # 6 -> 24, 224 -> 112
        self.down2 = DownBlock(in_channels*4, in_channels*16) # 24 -> 96, 112 -> 56
        self.down3 = DownBlock(in_channels*16, in_channels*64) # 96 -> 384, 56 -> 28
        self.down4 = DownBlock(in_channels*64, unet_encoder_size) # 384 -> 768, 28 -> 14
        
        # Bottleneck
        self.bottleneck = Block(unet_encoder_size, unet_encoder_size)
        
        # Decoder
        self.up1 = UpBlockWithSkip(2*unet_encoder_size, in_channels*64) # 768 -> 384, 14 -> 28
        self.up2 = UpBlockWithSkip(2*in_channels*64, in_channels*16) # 384 -> 96, 28 -> 56
        self.up3 = UpBlockWithSkip(2*in_channels*16, in_channels*4) # 96 -> 24, 56 -> 112
        self.up4 = UpBlockWithSkip(2*in_channels*4, in_channels) # 24 -> 6, 112 -> 224
        
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward_features(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
 
        return x

    def forward(self, s1_img, s2_img, dem_img, water_occur=None): # pass other modality for train loop compatiblity

        del s1_img
        del dem_img
        x = self.forward_features(s2_img)
        x = self.out(x)
        return x
    

def ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same',
               is_bn=True, is_relu=True, n=2):
    """ Custom function for conv2d:
        Apply 3*3 convolutions with BN and ReLU.
    """
    layers = []
    for i in range(1, n + 1):
        conv = nn.Conv2d(in_channels=in_channels if i == 1 else out_channels, 
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding if padding != 'same' else 'same',
                         bias=not is_bn)  # Disable bias when using BatchNorm
        layers.append(conv)
        
        if is_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if is_relu:
            layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)



