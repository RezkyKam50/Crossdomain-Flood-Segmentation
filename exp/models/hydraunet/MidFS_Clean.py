from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn


class DSUNetMidFS(nn.Module):

    def __init__(self, cfg, fusion="cat"):
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

        bottleneck_dim = topology[-1]

        self.fusion = fusion
        if fusion == "cat":
            self.middle_fusion_proj = nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, kernel_size=1)
        self.out_conv = OutConv(2 * topology[0], out)

    def forward(self, s1_img, s2_img, dem, pw):

        s1 = torch.cat([s1_img, dem, pw], dim=1)
        s2 = s2_img

        s1_skips = self.s1_stream.encode(s1)
        s2_skips = self.s2_stream.encode(s2)

        if self.fusion == "cat":
            fused_bottleneck = self.middle_fusion_proj(
                torch.cat([s1_skips[-1], s2_skips[-1]], dim=1)
            )
        elif self.fusion == "add":
            fused_bottleneck = s1_skips[-1] + s2_skips[-1]

        s1_skips[-1] = fused_bottleneck
        s2_skips[-1] = fused_bottleneck

        s1_feature = self.s1_stream.decode(s1_skips)
        s2_feature = self.s2_stream.decode(s2_skips)

        return self.out_conv(torch.cat([s1_feature, s2_feature], dim=1))


class UNet(nn.Module):

    def __init__(self, cfg, n_channels=None, n_classes=None, topology=None, enable_outc=True, weak=None):
        super(UNet, self).__init__()
        self._cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, DoubleConv if not weak else WeakDoubleConv)
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
            down_dict[f'down{idx + 1}'] = Down(in_dim, out_dim, DoubleConv if not weak else WeakDoubleConv)
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            up_dict[f'up{idx + 1}'] = Up(in_dim, out_dim, DoubleConv if not weak else WeakDoubleConv)
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


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class WeakDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(WeakDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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