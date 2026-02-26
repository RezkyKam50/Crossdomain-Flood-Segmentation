"""
Adapted from: https://github.com/javiribera/locating-objects-without-bboxes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.evanet.elev_conv import *
from models.evanet.elev_deconv import *


class double_conv(nn.Module):
    
    def __init__(self, img_in_ch, elev_in_ch, out_ch, normalize=True, activate=True):
        
        super(double_conv, self).__init__()
        
        self.first_Conv = ElevationConv(img_in_ch, elev_in_ch, out_ch)
        self.sec_conv = ElevationConv(out_ch, out_ch, out_ch)
        self.normalize = normalize
        self.activate = activate
        
        if self.normalize:
            self.first_batch_norm_img = nn.BatchNorm2d(out_ch)
            self.first_batch_norm_elev = nn.BatchNorm2d(out_ch)
            self.sec_batch_norm_img = nn.BatchNorm2d(out_ch)
            self.sec_batch_norm_elev = nn.BatchNorm2d(out_ch)
        if self.activate:
            self.first_activation_img = nn.ReLU()
            self.first_activation_elev = nn.ReLU()
            self.sec_activation_img = nn.ReLU()
            self.sec_activation_elev = nn.ReLU()

    def forward(self, x, h):

        x, h = self.first_Conv(x, h)
        if self.normalize:
            x = self.first_batch_norm_img(x)
            h = self.first_batch_norm_elev(h)
        if self.activate:
            x = self.first_activation_img(x)
            h = self.first_activation_elev(h)

        x, h = self.sec_conv(x, h)
        if self.normalize:
            x = self.sec_batch_norm_img(x)
            h = self.sec_batch_norm_elev(h)
        if self.activate:
            x = self.sec_activation_img(x)
            h = self.sec_activation_elev(h)

        return x, h


class inconv(nn.Module):
    def __init__(self, img_in_ch, elev_in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(img_in_ch, elev_in_ch, out_ch)

    def forward(self, x, h):
        x, h = self.conv(x, h)
        return x, h


class down(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=True):
        super(down, self).__init__()
        self.x_pool = nn.MaxPool2d(2)
        self.h_pool = nn.AvgPool2d(2)
        self.conv = double_conv(in_ch, in_ch, out_ch, normalize=normaliz)

    def forward(self, x, h):
        x = self.x_pool(x)
        h = self.h_pool(h)
        x, h = self.conv(x, h)
        return x, h


class up(nn.Module):
    
    def __init__(self, in_ch, cat_ch, out_ch, normalize=True, activate=True):
        super(up, self).__init__()

        self.normalize = normalize
        self.activate = activate
        
        self.up_conv = ElevationConvTranspose(in_ch, in_ch)
        self.conv = double_conv(cat_ch, cat_ch, out_ch, normalize=normalize, activate=activate)
        
        if self.normalize:
            self.first_batch_norm_img = nn.BatchNorm2d(in_ch)
            self.first_batch_norm_elev = nn.BatchNorm2d(in_ch)
        if self.activate:
            self.first_activation_img = nn.ReLU()
            self.first_activation_elev = nn.ReLU()

    def forward(self, x1, x2, h1, h2):
        
        ## Upsample with elevation transpose conv
        x1, h1 = self.up_conv(x1, h1)
        
        if self.normalize:
            x1 = self.first_batch_norm_img(x1)
            h1 = self.first_batch_norm_elev(h1)
        if self.activate:
            x1 = self.first_activation_img(x1)
            h1 = self.first_activation_elev(h1)

        ## Pad x1/h1 to match skip connection spatial dims (handles odd input sizes)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        if diff_h != 0 or diff_w != 0:
            x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                            diff_h // 2, diff_h - diff_h // 2])
            h1 = F.pad(h1, [diff_w // 2, diff_w - diff_w // 2,
                            diff_h // 2, diff_h - diff_h // 2])
        
        ## Concat upsampled output with skip connection
        x = torch.cat([x2, x1], dim=1)
        h = torch.cat([h2, h1], dim=1)
        # print("x_cat: ", x.shape)
        # print("h_cat: ", h.shape)
        
        ## Use an elev conv layer on merged output
        x, h = self.conv(x, h)
        
        return x, h


class outconv(nn.Module):
    def __init__(self, img_in_ch, elev_in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = ElevationConv(img_in_ch, elev_in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x, h):
        x, h = self.conv(x, h)
        return x


"""
Adapted from: https://github.com/javiribera/locating-objects-without-bboxes
"""