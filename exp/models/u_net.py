import torch
import torch.nn as nn 
from torch import nn
import torch
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = Block(in_channels, in_channels)
        self.downscale = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.block(x)
        x = self.downscale(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class UpBlockWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlockWithSkip, self).__init__()
        self.block = Block(in_channels, in_channels)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        x = self.up_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.block = Block(in_channels, in_channels)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.block(x)
        x = self.up_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

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
        
    def forward(self, s1_img, s2_img, dem, pw):
        x = s2_img
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

    def forward(self, x, x2, x3, x4): # pass other modality for train loop compatiblity
 
        x = self.forward_features(x)
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

def dot_product(seg, cls):
    b, n, h, w = seg.shape
    seg = seg.view(b, n, -1)
    cls = cls.unsqueeze(-1)  # Add an extra dimension for broadcasting
    final = torch.einsum("bik,bi->bik", seg, cls)
    final = final.view(b, n, h, w)
    return final

