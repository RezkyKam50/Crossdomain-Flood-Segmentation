from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.hydraunet.Attention import *

class EarlyFusionUNet(nn.Module):

    def __init__(self, cfg):
        super(EarlyFusionUNet, self).__init__()

 
        n_s2_bands = len(cfg.DATASET.SENTINEL2_BANDS)
        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        self.unet = UNet(
            n_channels=n_s2_bands + 2,
            n_classes=out,
            topology=topology,
        )

    def forward(self, s1_img=None, s2_img=None, dem=None, pw=None):
        fused_input = torch.cat([s2_img, dem, pw], dim=1)
        return self.unet(fused_input)


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, topology):
        super(UNet, self).__init__()

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, DoubleConv)
        self.outc = OutConv(first_chan, n_classes)

        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan]
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]
            down_dict[f'down{idx + 1}'] = Down(in_dim, out_dim, DoubleConv)
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            up_dict[f'up{idx + 1}'] = Up(in_dim, out_dim, DoubleConv)
        self.up_seq = nn.ModuleDict(up_dict)

    def encode(self, x):
        x1 = self.inc(x)
        inputs = [x1]
        for layer in self.down_seq.values():
            inputs.append(layer(inputs[-1]))
        return inputs

    def decode(self, inputs):
        inputs = list(inputs)
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x1 = layer(x1, inputs[idx])
        return self.outc(x1)

    def forward(self, x):
        return self.decode(self.encode(x))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.projection = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.projection(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch),
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

        # self.attn_scheme = False
        # self.skip_attn = CoordAtt(in_ch // 2, in_ch // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
 
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
        ])

        # if self.attn_scheme:
        #     x2 = self.skip_attn(x2) 
            
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x