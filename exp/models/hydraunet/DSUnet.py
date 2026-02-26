from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from models.prithvi_encoder import PrithviEncoder

# Dual Stream Classical UNet
# Reference from DS_Unet https://github.com/SebastianHafner/DS_UNet/blob/master/utils/networks.py


class DSUNet(nn.Module):

    def __init__(self, cfg, use_prithvi=None, use_cm_attn=None, fusion_scheme="late", bottleneck_dropout_prob=None):
        super(DSUNet, self).__init__()
        self._cfg = cfg
        self.fusion_scheme = fusion_scheme
        self.use_prithvi = use_prithvi
        self.use_attention = use_cm_attn

        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        n_s1_bands = len(cfg.DATASET.SENTINEL1_BANDS)
        n_s2_bands = len(cfg.DATASET.SENTINEL2_BANDS)
        self.n_s1_bands = n_s1_bands
        self.n_s2_bands = n_s2_bands

        if fusion_scheme == "early":
 
            self.fused_stream = UNet(
                cfg,
                n_channels=n_s1_bands + n_s2_bands,
                n_classes=out,
                topology=topology,
                enable_outc=True,
                bottleneck_dropout_prob=bottleneck_dropout_prob
            )
            if use_prithvi:
                self.prithvi = PrithviEncoder(
                    weights_path=cfg.MODEL.PRITHVI_PATH,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    target_channels=cfg.MODEL.TOPOLOGY[-1]
                )
 

        elif fusion_scheme == "late":
 
            self.s1_stream = UNet(cfg, n_channels=n_s1_bands, n_classes=out,
                                  topology=topology, enable_outc=False,
                                  bottleneck_dropout_prob=bottleneck_dropout_prob)
            self.s2_stream = UNet(cfg, n_channels=n_s2_bands, n_classes=out,
                                  topology=topology, enable_outc=False,
                                  bottleneck_dropout_prob=bottleneck_dropout_prob)
            # if self.use_attention:
            #     self.cross_attn = CrossModalFusion(embed_dim=topology[0])
            if self.use_prithvi:
                self.prithvi = PrithviEncoder(
                    weights_path=cfg.MODEL.PRITHVI_PATH,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    target_channels=cfg.MODEL.TOPOLOGY[-1]
                )
            self.out_conv = OutConv(2 * topology[0], out)

        elif fusion_scheme == "middle":
 
            self.s1_stream = UNet(cfg, n_channels=n_s1_bands, n_classes=out,
                                  topology=topology, enable_outc=False,
                                  bottleneck_dropout_prob=bottleneck_dropout_prob)
            self.s2_stream = UNet(cfg, n_channels=n_s2_bands, n_classes=out,
                                  topology=topology, enable_outc=False,
                                  bottleneck_dropout_prob=bottleneck_dropout_prob)
            # if self.use_attention:
            #     self.cross_attn = CrossModalFusion(embed_dim=topology[-1])
            if self.use_prithvi:
                self.prithvi = PrithviEncoder(
                    weights_path=cfg.MODEL.PRITHVI_PATH,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    target_channels=cfg.MODEL.TOPOLOGY[-1]
                )

            bottleneck_dim = topology[-1]
 
            self.middle_fusion_proj = nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, kernel_size=1)
            self.out_conv = OutConv(2 * topology[0], out)

        else:
            raise ValueError(f"Unknown fusion_scheme: {fusion_scheme}. "
                             f"Choose from 'early', 'middle', 'late'.")
 

    def change_prithvi_trainability(self, trainable):
        if hasattr(self, 'prithvi'):
            for param in self.prithvi.prithvi.parameters():
                param.requires_grad = trainable

    def change_s1_trainability(self, trainable):
        if hasattr(self, 's1_stream'):
            for param in self.s1_stream.parameters():
                param.requires_grad = trainable

    def change_s2_trainability(self, trainable):
        if hasattr(self, 's2_stream'):
            for param in self.s2_stream.parameters():
                param.requires_grad = trainable
 

    def forward(self, s1_img, s2_img, dem_img):
        del dem_img

        if self.fusion_scheme == "early":
            fused_input = torch.cat([s1_img, s2_img], dim=1)
            if self.use_prithvi:
                prithvi_features = self.prithvi(s2_img)
                return self.fused_stream(fused_input, prithvi_features=prithvi_features)
            return self.fused_stream(fused_input)

        elif self.fusion_scheme == "late":
            s1_feature = self.s1_stream(s1_img)

            if self.use_prithvi:
                prithvi_features = self.prithvi(s2_img)
                s2_feature = self.s2_stream(s2_img, prithvi_features=prithvi_features)
            else:
                s2_feature = self.s2_stream(s2_img)

            if self.use_attention:
 
                s1_feature, s2_feature = self.cross_attn(s1_feature, s2_feature)

            fusion = torch.cat((s1_feature, s2_feature), dim=1)
            return self.out_conv(fusion)

        elif self.fusion_scheme == "middle":
            s1_skips = self.s1_stream.encode(s1_img)
            s2_skips = self.s2_stream.encode(s2_img)

            if self.use_attention:
 
                s1_bot, s2_bot = self.cross_attn(s1_skips[-1], s2_skips[-1])
                fused_bottleneck = self.middle_fusion_proj(torch.cat([s1_bot, s2_bot], dim=1))
            else:
                fused_bottleneck = self.middle_fusion_proj(
                    torch.cat([s1_skips[-1], s2_skips[-1]], dim=1)
                )

            if self.use_prithvi:
                prithvi_features = self.prithvi(s2_img)
                fused_bottleneck = self.s2_stream._combine(fused_bottleneck, prithvi_features)

            s1_skips[-1] = fused_bottleneck
            s2_skips[-1] = fused_bottleneck

            s1_feature = self.s1_stream.decode(s1_skips)
            s2_feature = self.s2_stream.decode(s2_skips)

            fusion = torch.cat([s1_feature, s2_feature], dim=1)
            return self.out_conv(fusion)

 

class UNet(nn.Module):

    def __init__(self, cfg, n_channels=None, n_classes=None, topology=None,
                 enable_outc=True, combine_method=None, bottleneck_dropout_prob=None):

        self._cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        super(UNet, self).__init__()

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, DoubleConv)
        self.enable_outc = enable_outc
        self.outc = OutConv(first_chan, n_classes)

        if combine_method:
            self.bottleneck_dropout = RandomHalfDropoutLayer(bottleneck_dropout_prob)
        else:
            self.bottleneck_dropout = None

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
 

    def _combine(self, x_unet, x_prithvi):
        if x_prithvi.shape[2:] != x_unet.shape[2:]:
            x_prithvi = F.interpolate(x_prithvi, size=x_unet.shape[2:],
                                      mode='bilinear', align_corners=False)
        if self.combine_method == 'concat':
            fused = torch.cat([x_unet, x_prithvi], dim=1)
            return self.bottleneck_proj(fused)
        elif self.combine_method == 'add':
            return x_unet + x_prithvi
        elif self.combine_method == 'mul':
            return x_unet * x_prithvi
 

    def encode(self, x):
 
        x1 = self.inc(x)
        inputs = [x1]
        for idx, layer in enumerate(self.down_seq.values()):
            out = layer(inputs[-1])
            if idx == len(self.down_seq) - 1 and self.combine_method is not None:   # bottleneck
                out = self.bottleneck_dropout(out)
            inputs.append(out)
        return inputs

    def decode(self, inputs, prithvi_features=None):
 
        inputs = list(inputs)  # avoid mutating caller's list

        if prithvi_features is not None:
            inputs[-1] = self._combine(inputs[-1], prithvi_features)

        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x1 = layer(x1, x2)

        return self.outc(x1) if self.enable_outc else x1
 

    def forward(self, x1, x2=None, x3=None, prithvi_features=None):
        if x2 is None and x3 is None:
            x = x1
        elif x3 is None:
            x = torch.cat((x1, x2), 1)
        else:
            x = torch.cat((x1, x2, x3), 1)

        skips = self.encode(x)
        return self.decode(skips, prithvi_features=prithvi_features)

 
 
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True)
        )
 
    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
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




 

class RandomHalfDropoutLayer(nn.Module):
    def __init__(self, dropout_prob=None):
        self.dropout_prob = dropout_prob
        super(RandomHalfDropoutLayer, self).__init__()

    def forward(self, x):
        if not self.training or self.dropout_prob == 0:
            return x  # No dropout in evaluation mode
        
        batch_size, num_channels, _, _ = x.size()
        half_channels = num_channels // 2
        
        # Create a random strategy tensor for the whole batch (0: mask upper half, 1: mask lower half, 2: no masking)
        strategies = torch.empty(batch_size, 1, 1, 1, device=x.device).uniform_(0, 1)
        
        # Adjust strategies based on dropout probability
        mask_prob = self.dropout_prob / 2
        strategies = torch.where(strategies < mask_prob, torch.tensor(0, device=x.device), strategies)
        strategies = torch.where((strategies >= mask_prob) & (strategies < 2 * mask_prob), torch.tensor(1, device=x.device), strategies)
        strategies = torch.where(strategies >= 2 * mask_prob, torch.tensor(2, device=x.device), strategies)
        
        # Create masks for upper and lower halves
        upper_mask = torch.ones_like(x)
        lower_mask = torch.ones_like(x)
        
        # Mask the upper half
        upper_mask[:, :half_channels, :, :] = 0
        # Mask the lower half
        lower_mask[:, half_channels:, :, :] = 0
        
        # Apply the strategies to create a final mask and scaling
        # Case 0: Apply upper mask and scale lower half by 2
        x = torch.where(strategies == 0, x * lower_mask * 2, x)
        
        # Case 1: Apply lower mask and scale upper half by 2
        x = torch.where(strategies == 1, x * upper_mask * 2, x)
        
        # Case 2: No masking (leave x as it is)
        return x