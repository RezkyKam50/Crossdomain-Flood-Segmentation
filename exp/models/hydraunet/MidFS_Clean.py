from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=None):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.sharedMLP = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))

#     def forward(self, x):
#         avgout = self.sharedMLP(self.avg_pool(x))
#         maxout = self.sharedMLP(self.max_pool(x))
#         return F.softmax(avgout + maxout, dim=1)

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        self.attn = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # flatten spatial dims -> (B, H*W, C)
        x_flat = self.norm(x).view(B, C, -1).permute(0, 2, 1)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        # reshape back + residual
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return x + attn_out

class FusionProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1, bias=False),
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
        # self.s1_attn = ChannelAttention(bottleneck_dim, 4)
        # self.s2_attn = ChannelAttention(bottleneck_dim, 4)

        self.dp_s1 = DropBlock2D(drop_prob=0.15, block_size=7)
        self.dp_s2 = DropBlock2D(drop_prob=0.15, block_size=7)
        self.s1_attn = SelfAttention2D(bottleneck_dim, num_heads=4)
        self.s2_attn = SelfAttention2D(bottleneck_dim, num_heads=4)

        self.fusion_weight = nn.Parameter(torch.ones(2) / 2)

        self.out_conv = OutConv(topology[0], out)

    def forward(self, s1_img, s2_img, dem, pw):
        s1 = torch.cat([s1_img, dem, pw], dim=1)
        s1_skips = self.s1_stream.encode(s1)
        s2_skips = self.s2_stream.encode(s2_img)

        fused = self.bottleneck_fusion(s1_skips[-1], s2_skips[-1])
        s1_skips[-1] = self.dp_s1(self.s1_attn(s1_skips[-1] + fused))
        s2_skips[-1] = self.dp_s2(self.s2_attn(s2_skips[-1] + fused))

        s1_feature = self.s1_stream.decode(s1_skips)
        s2_feature = self.s2_stream.decode(s2_skips)

        w = torch.softmax(self.fusion_weight, dim=0)
        s1_feature = (w[0] * s1_feature) 
        s2_feature = (w[1] * s2_feature) 
        
        combined = s1_feature + s2_feature

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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return out


class WeakConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
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