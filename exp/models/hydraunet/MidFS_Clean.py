from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class DropPath(nn.Module):
    ''' Stochastic Dropout  (Gao Huang et al) applied to modules with residual connection'''
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def drop_path(self, x: Tensor, keep_prob: float = 1.0, inplace: bool = False) -> Tensor:
        mask_shape: Tuple[int] = (x.shape[0],) + (1,) * (x.ndim - 1) 
        # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
        mask: Tensor = x.new_empty(mask_shape).bernoulli_(keep_prob)
        mask.div_(keep_prob)
        if inplace:
            x.mul_(mask)
        else:
            x = x * mask
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0:
            x = self.drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})" 
 

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=None):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return F.softmax(avgout + maxout, dim=1)

class ModalityGate(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
  
        self.gate = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim * 2, 1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, 2, 1),  
            nn.Softmax(dim=1)  
        )
 
        self.proj = nn.Conv2d(feature_dim, feature_dim * 2, 1)
    
    def forward(self, s1_feat, s2_feat):
        combined = torch.cat([s1_feat, s2_feat], dim=1)
        gates = self.gate(combined)  
        s1_gate = gates[:, 0:1, :, :]   
        s2_gate = gates[:, 1:2, :, :]  
         
        gated = s1_feat * s1_gate + s2_feat * s2_gate   
        return self.proj(gated)  

class FusionProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        return self.proj(torch.cat([x1, x2], dim=1))


class DSUNetMidFS(nn.Module):
    def __init__(self, cfg):
        super(DSUNetMidFS, self).__init__()
        self._cfg = cfg

        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY
        n_s1_bands = len(cfg.DATASET.SENTINEL1_BANDS)
        n_s2_bands = len(cfg.DATASET.SENTINEL2_BANDS)

        self.s1_stream = UNet(cfg, n_channels=n_s1_bands + 2, n_classes=out,
                              topology=topology, enable_outc=False, weak=True)

        self.s2_stream = UNet(cfg, n_channels=n_s2_bands, n_classes=out,
                              topology=topology, enable_outc=False, weak=False)

        # Bottleneck dim is the last topology value
        bottleneck_dim = topology[-1]

        self.bottleneck_fusion = FusionProjection(bottleneck_dim)
        self.s1_attn = ChannelAttention(bottleneck_dim, 4)
        self.s2_attn = ChannelAttention(bottleneck_dim, 4)
        self.modality_gate = ModalityGate(topology[0])

        self.out_conv = OutConv(2 * topology[0], out)

    def forward(self, s1_img, s2_img, dem, pw):
        s1 = torch.cat([s1_img, dem, pw], dim=1)
        s1_skips = self.s1_stream.encode(s1)
        s2_skips = self.s2_stream.encode(s2_img)

        # Only fuse at the bottleneck (last element of skips)
        fused = self.bottleneck_fusion(s1_skips[-1], s2_skips[-1])
        s1_skips[-1] = s1_skips[-1] * self.s1_attn(fused)
        s2_skips[-1] = s2_skips[-1] * self.s2_attn(fused)

        s1_feature = self.s1_stream.decode(s1_skips)
        s2_feature = self.s2_stream.decode(s2_skips)

        # combined = torch.cat([s1_feature, s2_feature], dim=1)

        combined = self.modality_gate(s1_feature, s2_feature)

        return self.out_conv(combined)


class UNet(nn.Module):

    def __init__(self, cfg, n_channels=None, n_classes=None, topology=None, enable_outc=True, weak=None):
        super(UNet, self).__init__()
        self._cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, ConvBlock if not weak else WeakConvBlock)
        self.enable_outc = enable_outc
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
            down_dict[f'down{idx + 1}'] = Down(in_dim, out_dim, ConvBlock if not weak else WeakConvBlock)
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            up_dict[f'up{idx + 1}'] = Up(in_dim, out_dim, ConvBlock if not weak else WeakConvBlock)
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
        return self.outc(x1) if self.enable_outc else x1

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(p, inplace=False) if p > 0 else nn.Identity()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.drop_path(out)
        out = out + residual
        out = self.act(out)
        return out


class WeakConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
        )

        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(p, inplace=False) if p > 0 else nn.Identity()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.drop_path(out)
        out = out + residual
        out = self.act(out)
        return out


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
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))