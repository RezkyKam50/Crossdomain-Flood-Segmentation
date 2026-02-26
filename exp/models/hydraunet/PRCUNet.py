from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.hydraunet.PRCNPTN import PRCNPTNLayer
 
# Dual Stream Classical UNet

# Reference from DS_Unet https://github.com/SebastianHafner/DS_UNet/blob/master/utils/networks.py
class DSUNet_PRC(nn.Module):
    def __init__(self, cfg):
        super(DSUNet_PRC, self).__init__()
        assert (cfg.DATASET.MODE == 'fusion')
        self._cfg = cfg
        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        # self.CBAM_S1 = CBAM(topology[0])
        # self.CBAM_S2 = CBAM(topology[0])

        n_s1_bands = len(cfg.DATASET.SENTINEL1_BANDS)
        n_s2_bands = len(cfg.DATASET.SENTINEL2_BANDS)
        self.n_s1_bands = n_s1_bands
        self.n_s2_bands = n_s2_bands
 

        self.s1_stream = UNet(cfg, n_channels=n_s1_bands, n_classes=out,
                                topology=topology, enable_outc=False)
        self.s2_stream = UNet(cfg, n_channels=n_s2_bands, n_classes=out,
                                topology=topology, enable_outc=False,)
         
        self.out_conv = OutConv(2 * topology[0], out)

    def change_prithvi_trainability(self, trainable):
        if self.use_prithvi:
            self.prithvi.change_prithvi_trainability(trainable)

    def forward(self, s1_img, s2_img, dem_img): # pass dem modality for train loop compatiblity

        del dem_img

        
        s1_feature = self.s1_stream(s1_img)
        s2_feature = self.s2_stream(s2_img)

        # s1_feature = self.CBAM_S1(s1_feature)
        # s2_feature = self.CBAM_S1(s2_feature)


        fusion = torch.cat((s1_feature, s2_feature), dim=1)
        return self.out_conv(fusion)



class UNet(nn.Module):

    def __init__(self, cfg, n_channels=None, n_classes=None, topology=None,
                 enable_outc=True, combine_method=None):

        self._cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        super(UNet, self).__init__()

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, DoubleConv)
        self.enable_outc = enable_outc
        self.outc = OutConv(first_chan, n_classes)
 
        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan]
        up_dict = OrderedDict()

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]

            layer = Down(in_dim, out_dim, DoubleConv)
            print(f'down{idx + 1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx + 1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]

            layer = Up(in_dim, out_dim, DoubleConv)
            print(f'up{idx + 1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)
 
        self.combine_method = combine_method 
        bottleneck_dim = topology[-1]
        if combine_method == 'concat':
            self.bottleneck_proj = nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, kernel_size=1)
        else:
            self.bottleneck_proj = None
 

    def encode(self, x):
 
        x1 = self.inc(x)
        inputs = [x1]
        for idx, layer in enumerate(self.down_seq.values()):
            out = layer(inputs[-1])
            inputs.append(out)
        return inputs

    def decode(self, inputs):
 
        inputs = list(inputs)  # avoid mutating caller's list
 

        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x1 = layer(x1, x2)

        return self.outc(x1) if self.enable_outc else x1
 

    def forward(self, x1, x2=None, x3=None):
        if x2 is None and x3 is None:
            x = x1
        elif x3 is None:
            x = torch.cat((x1, x2), 1)
        else:
            x = torch.cat((x1, x2, x3), 1)

        skips = self.encode(x)
        return self.decode(skips)





# sub-parts of the U-Net model
 
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, G=8, CMP=2):
        super(DoubleConv, self).__init__()
          
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_ch)
        self.prc = PRCNPTNLayer(
            inch=out_ch,    
            outch=out_ch,
            G=G,
            CMP=CMP,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn_proj(self.proj(x))   # project in_ch -> out_ch
        x = self.prc(x)
        x = self.act2(self.bn2(x))
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle padding for 2D images
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
        ])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1)
        )

    def forward(self, x):
        x = self.projection(x)
        return x
    



