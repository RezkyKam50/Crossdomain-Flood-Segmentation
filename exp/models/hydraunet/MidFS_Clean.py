from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Tuple

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



class CrossModalAttention(nn.Module):
    """
    Each modality queries the other.
    x1 (query) attends to x2 (key/value) -> out1 = x1 + cross_attn(x1, x2)
    x2 (query) attends to x1 (key/value) -> out2 = x2 + cross_attn(x2, x1)
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)
        # shared attention for both directions
        self.attn_1to2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_2to1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def _flatten(self, x: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        return x.view(B, C, -1).permute(0, 2, 1)

    def _restore(self, x_flat: Tensor, shape: Tuple[int, ...]) -> Tensor:
        # (B, H*W, C) -> (B, C, H, W)
        B, C, H, W = shape
        return x_flat.permute(0, 2, 1).view(B, C, H, W)

    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
        shape1, shape2 = x1.shape, x2.shape

        q1 = self._flatten(self.norm1(x1))
        kv2 = self._flatten(self.norm2(x2))
        q2 = self._flatten(self.norm2(x2))
        kv1 = self._flatten(self.norm1(x1))

        # S1 queries S2
        out1, _ = self.attn_1to2(q1, kv2, kv2)
        # S2 queries S1
        out2, _ = self.attn_2to1(q2, kv1, kv1)

        return (
            x1 + self._restore(out1, shape1),
            x2 + self._restore(out2, shape2),
        )

class GradientFilter(nn.Module):
    def __init__(self, in_channels, branch_channels, num_directions=8, kernel_size=3):
        super().__init__()
        self.num_directions = num_directions
        self.branches = nn.ModuleList()
        angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5][:num_directions] 
        for angle in angles:
            branch = nn.Conv2d(in_channels, branch_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
            weight = self.get_rotated_sobel_kernel(angle, in_channels, branch_channels, kernel_size)
            branch.weight.data.copy_(weight)
            self.branches.append(branch)
        
        self.fusion_conv = nn.Conv2d(branch_channels * num_directions, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)

    def get_rotated_sobel_kernel(self, angle, in_channels, out_channels, kernel_size):
        base_kernel = torch.tensor([[-1., 0., 1.],
                                     [-2., 0., 2.],
                                     [-1., 0., 1.]], dtype=torch.float32)
        theta = math.radians(angle)
        rotated = torch.zeros_like(base_kernel)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = j - center
                y = i - center
                xr = x * math.cos(theta) - y * math.sin(theta)
                yr = x * math.sin(theta) + y * math.cos(theta)
                xi = int(round(xr)) + center
                yi = int(round(yr)) + center
                if 0 <= xi < kernel_size and 0 <= yi < kernel_size:
                    rotated[i, j] = base_kernel[yi, xi]

        weight = rotated.unsqueeze(0).unsqueeze(0).repeat(out_channels, in_channels, 1, 1)
        return weight

    def forward(self, x):
        branch_outs = []
        for branch in self.branches:
            out = branch(x)
            out = F.relu(out)
            branch_outs.append(out)
        multi_grad = torch.cat(branch_outs, dim=1)
        fused = self.fusion_conv(multi_grad)
        fused = self.norm(fused)
        return fused
 
class SC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.norm(self.pw(self.dw(x)))
    
class FGE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.sc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            SC(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.md_gradient = GradientFilter(
            in_channels, branch_channels=in_channels // 2, num_directions=8
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        self.dilated_conv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=d, padding=d)
            for d in [1, 2, 3]
        ])

        self.fusion_multi_scale = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(in_channels, eps=1e-6)

        self.gaussian = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        nn.init.normal_(self.gaussian.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x = x.float()

        feat = self.sc(x) + x
        
        grad_feat = self.md_gradient(feat)
        fused_feat = feat + grad_feat

        ca = self.channel_att(fused_feat)
        sa = self.spatial_att(fused_feat)
        att_feat = fused_feat * ca * sa + fused_feat

        H, W = att_feat.shape[2:]

        ms_features = []
        for conv in self.dilated_conv:
            feat_d = conv(att_feat)
            ms_features.append(F.interpolate(feat_d, size=(H, W), mode='bilinear', align_corners=False))

        ms_fused = self.fusion_multi_scale(torch.cat(ms_features, dim=1))

        normed = self.norm(ms_fused)
        filtered = self.gaussian(normed)

        output = filtered + att_feat
        return output



class SatelliteSTN(nn.Module):
    def __init__(self, s1_channels, s2_channels, feat_dim):
        super().__init__()
        '''Predicts x,y shifts, resamples pixels'''
        self.s1_proj = nn.Sequential(nn.Conv2d(s1_channels, feat_dim, 1), nn.ReLU())
        self.s2_proj = nn.Sequential(nn.Conv2d(s2_channels, feat_dim, 1), nn.ReLU())

        self.fge_s1 = FGE(feat_dim)
        self.fge_s2 = FGE(feat_dim)

        fused_ch = feat_dim * 2

        self.coarse_loc = nn.Sequential(
            nn.Conv2d(fused_ch, 32, 7, padding=3),
            nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.AdaptiveAvgPool2d(4), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 6)
        )
        self.coarse_loc[-1].weight.data.zero_()
        self.coarse_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fine_loc = nn.Sequential(
            nn.Conv2d(fused_ch, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, s1, s2):
        s1_enh = self.fge_s1(self.s1_proj(s1))
        s2_enh = self.fge_s2(self.s2_proj(s2))

        fused = torch.cat([s1_enh, s2_enh], dim=1)
        theta = self.coarse_loc(fused).view(-1, 2, 3)
        grid_c = F.affine_grid(theta, s1.size(), align_corners=False)
        s1_coarse = F.grid_sample(s1, grid_c, align_corners=False,
                                   padding_mode='reflection')

        s1_coarse_enh = self.fge_s1(self.s1_proj(s1_coarse))
        fused2 = torch.cat([s1_coarse_enh, s2_enh], dim=1)
        flow = self.fine_loc(fused2) * 0.1

        B, _, H, W = s1.shape
        base = F.affine_grid(
            torch.eye(2, 3, device=s1.device).unsqueeze(0).expand(B, -1, -1),
            s1.size(), align_corners=False)
        grid_f = base + flow.permute(0, 2, 3, 1)
        s1_aligned = F.grid_sample(s1_coarse, grid_f, align_corners=False,
                                    padding_mode='reflection')

        return s1_aligned
    

class DSUNetMidFS(nn.Module):
    def __init__(self, cfg):
        super(DSUNetMidFS, self).__init__()
        self._cfg = cfg

        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY
        n_s1_bands = len(cfg.DATASET.SENTINEL1_BANDS) + 2 # dem, pw each 1 ch
        n_s2_bands = len(cfg.DATASET.SENTINEL2_BANDS)
 
        self.s1_stream = UNet(cfg, n_channels=n_s1_bands, n_classes=out,
                              topology=topology, enable_outc=False, weak=True)
        self.s2_stream = UNet(cfg, n_channels=n_s2_bands, n_classes=out,
                              topology=topology, enable_outc=False, weak=False)

        bottleneck_dim = topology[-1]

        self.bottleneck_cma = CrossModalAttention(bottleneck_dim, num_heads=8)

        self.s1_aligner = SatelliteSTN(n_s1_bands, n_s2_bands, feat_dim=topology[0])

        self.fusion_weight = nn.Parameter(torch.ones(2, topology[0]) / 2)
        self.out_conv = OutConv(topology[0], out)


    def forward(self, s1_img, s2_img, dem, pw):
        s1_img = torch.cat([s1_img, dem, pw], dim=1)
        s1_img = self.s1_aligner(s1_img, s2_img)

        s1_skips = self.s1_stream.encode(s1_img)
        s2_skips = self.s2_stream.encode(s2_img)

        s1_bot, s2_bot = self.bottleneck_cma(s1_skips[-1], s2_skips[-1])

        s1_skips[-1] = s1_bot
        s2_skips[-1] = s2_bot

        s1_feature = self.s1_stream.decode(s1_skips)
        s2_feature = self.s2_stream.decode(s2_skips)

        w = torch.softmax(self.fusion_weight, dim=0)
        combined = (w[0].view(1, -1, 1, 1) * s1_feature) + (w[1].view(1, -1, 1, 1) * s2_feature)

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