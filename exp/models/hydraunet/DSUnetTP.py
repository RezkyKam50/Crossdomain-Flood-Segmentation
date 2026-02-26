import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from models.prithvi_segmenter import PritviSegmenter
 
# Dual Stream UNet3+
# Reference from DS_Unet https://github.com/SebastianHafner/DS_UNet/blob/master/utils/networks.py


class DSUNet3P(nn.Module):
    def __init__(self, cfg, use_prithvi=None, use_cm_attn=None,
                 fusion_scheme="late", bottleneck_dropout_prob=2/3):
        super(DSUNet3P, self).__init__()
        assert (cfg.DATASET.MODE == 'fusion')
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
            # Single stream: concatenate S1+S2 before any processing
            self.fused_stream = UNet3Plus(
                cfg,
                n_channels=n_s1_bands + n_s2_bands,
                n_classes=out,
                enable_outc=True,
                bottleneck_dropout_prob=bottleneck_dropout_prob
            )
            if use_prithvi:
                self.prithvi = PritviSegmenter(
                    weights_path=cfg.MODEL.PRITHVI_PATH,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    prithvi_encoder_size=cfg.MODEL.TOPOLOGY[-1],
                    output_channels=out
                )
            # No out_conv needed — fused_stream handles it internally

        elif fusion_scheme == "late":
            # Two independent streams, fuse at output
            # cross_attn operates on topology[0] — the finest/largest feature map.
            self.s1_stream = UNet3Plus(cfg, n_channels=n_s1_bands, n_classes=out,
                                       enable_outc=False,
                                       bottleneck_dropout_prob=bottleneck_dropout_prob)
            self.s2_stream = UNet3Plus(cfg, n_channels=n_s2_bands, n_classes=out,
                                       enable_outc=False,
                                       bottleneck_dropout_prob=bottleneck_dropout_prob)
            # if self.use_attention:
            #     self.cross_attn = CrossModalFusion(embed_dim=topology[0])
            if use_prithvi:
                self.prithvi = PritviSegmenter(
                    weights_path=cfg.MODEL.PRITHVI_PATH,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    prithvi_encoder_size=cfg.MODEL.TOPOLOGY[-1],
                    output_channels=out
                )
                out_dim = 2 * topology[0] + 2  # prithvi adds 2 channels
            else:
                out_dim = 2 * topology[0]
            self.out_conv = OutConv(out_dim, out)

        elif fusion_scheme == "middle":
            # Two streams encode independently, fuse at bottleneck, decode separately
            # cross_attn operates on topology[-1] — the bottleneck (smallest spatial size).
            self.s1_stream = UNet3Plus(cfg, n_channels=n_s1_bands, n_classes=out,
                                       enable_outc=False,
                                       bottleneck_dropout_prob=bottleneck_dropout_prob)
            self.s2_stream = UNet3Plus(cfg, n_channels=n_s2_bands, n_classes=out,
                                       enable_outc=False,
                                       bottleneck_dropout_prob=bottleneck_dropout_prob)
            # if self.use_attention:
            #     self.cross_attn = CrossModalFusion(embed_dim=topology[-1])
            if use_prithvi:
                self.prithvi = PritviSegmenter(
                    weights_path=cfg.MODEL.PRITHVI_PATH,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    prithvi_encoder_size=cfg.MODEL.TOPOLOGY[-1],
                    output_channels=out
                )

            bottleneck_dim = topology[-1]
            # middle_fusion_proj fuses the two CA-refined bottlenecks into one
            self.middle_fusion_proj = nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, kernel_size=1)

            out_dim = 2 * topology[0] + 2 if use_prithvi else 2 * topology[0]
            self.out_conv = OutConv(out_dim, out)

        else:
            raise ValueError(f"Unknown fusion_scheme: {fusion_scheme}. "
                             f"Choose from 'early', 'middle', 'late'.")

    def change_prithvi_trainability(self, trainable):
        if self.use_prithvi:
            self.prithvi.change_prithvi_trainability(trainable)

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
                s2_feature = self.s2_stream(s2_img)
            else:
                s2_feature = self.s2_stream(s2_img)

            if self.use_attention:
                s1_feature, s2_feature = self.cross_attn(s1_feature, s2_feature)

            if self.use_prithvi:
                fusion = torch.cat((s1_feature, s2_feature, prithvi_features), dim=1)
            else:
                fusion = torch.cat((s1_feature, s2_feature), dim=1)

            return self.out_conv(fusion)

        elif self.fusion_scheme == "middle":
            s1_skips = self.s1_stream.encode(s1_img)
            s2_skips = self.s2_stream.encode(s2_img)

            if self.use_attention:
                # Refine both bottlenecks cross-modally, then concatenate + project
                s1_bot, s2_bot = self.cross_attn(s1_skips[-1], s2_skips[-1])
                fused_bottleneck = self.middle_fusion_proj(torch.cat([s1_bot, s2_bot], dim=1))
            else:
                fused_bottleneck = self.middle_fusion_proj(
                    torch.cat([s1_skips[-1], s2_skips[-1]], dim=1)
                )

            if self.use_prithvi:
                prithvi_features = self.prithvi(s2_img)
                # Fuse prithvi into the shared bottleneck via the s2 stream's _combine method
                fused_bottleneck = self.s2_stream._combine_prithvi(fused_bottleneck, prithvi_features)

            # Share the fused bottleneck across both decode paths
            s1_skips[-1] = fused_bottleneck
            s2_skips[-1] = fused_bottleneck

            s1_feature = self.s1_stream.decode(s1_skips)
            s2_feature = self.s2_stream.decode(s2_skips)

            fusion = torch.cat([s1_feature, s2_feature], dim=1)
            return self.out_conv(fusion)


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
        if act:
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = nn.Sequential(
            conv_block(in_c, out_c),
            conv_block(out_c, out_c)
        )
        self.p1 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.c1(x)
        p = self.p1(x)
        return x, p


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


# Ref https://github.com/nikhilroxtomar/UNET-3-plus-Implementation-in-TensorFlow-and-PyTorch/blob/main/pytorch/1-unet3plus.py
class UNet3Plus(nn.Module):
    def __init__(self, cfg, n_channels=None, n_classes=None, enable_outc=True,
                 bottleneck_dropout_prob=2/3):
        super().__init__()

        self._cfg = cfg
        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes

        if hasattr(cfg.MODEL, 'TOPOLOGY'):
            topology = cfg.MODEL.TOPOLOGY
            f1, f2, f3, f4, f5 = topology
        else:
            # Legacy UNet3+ Topo
            f1, f2, f3, f4, f5 = 64, 128, 256, 512, 1024

        self.enable_outc = enable_outc

        self.e1 = encoder_block(n_channels, f1)
        self.e2 = encoder_block(f1, f2)
        self.e3 = encoder_block(f2, f3)
        self.e4 = encoder_block(f3, f4)

        self.e5 = nn.Sequential(
            conv_block(f4, f5),
            conv_block(f5, f5)
        )

        self.bottleneck_dropout = RandomHalfDropoutLayer(dropout_prob=bottleneck_dropout_prob)

        # Prithvi bottleneck fusion
        self.bottleneck_proj = nn.Conv2d(f5 * 2, f5, kernel_size=1)

        self.reduction_channels = f1

        self.e1_d4 = conv_block(f1, self.reduction_channels)
        self.e2_d4 = conv_block(f2, self.reduction_channels)
        self.e3_d4 = conv_block(f3, self.reduction_channels)
        self.e4_d4 = conv_block(f4, self.reduction_channels)
        self.e5_d4 = conv_block(f5, self.reduction_channels)
        self.d4 = conv_block(self.reduction_channels * 5, self.reduction_channels)

        self.e1_d3 = conv_block(f1, self.reduction_channels)
        self.e2_d3 = conv_block(f2, self.reduction_channels)
        self.e3_d3 = conv_block(f3, self.reduction_channels)
        self.e4_d3 = conv_block(self.reduction_channels, self.reduction_channels)  # from d4
        self.e5_d3 = conv_block(f5, self.reduction_channels)
        self.d3 = conv_block(self.reduction_channels * 5, self.reduction_channels)

        self.e1_d2 = conv_block(f1, self.reduction_channels)
        self.e2_d2 = conv_block(f2, self.reduction_channels)
        self.e3_d2 = conv_block(self.reduction_channels, self.reduction_channels)  # from d3
        self.e4_d2 = conv_block(self.reduction_channels, self.reduction_channels)  # from d4
        self.e5_d2 = conv_block(f5, self.reduction_channels)
        self.d2 = conv_block(self.reduction_channels * 5, self.reduction_channels)

        self.e1_d1 = conv_block(f1, self.reduction_channels)
        self.e2_d1 = conv_block(self.reduction_channels, self.reduction_channels)  # from d2
        self.e3_d1 = conv_block(self.reduction_channels, self.reduction_channels)  # from d3
        self.e4_d1 = conv_block(self.reduction_channels, self.reduction_channels)  # from d4
        self.e5_d1 = conv_block(f5, self.reduction_channels)
        self.d1 = conv_block(self.reduction_channels * 5, self.reduction_channels)

        if enable_outc:
            self.y1 = nn.Conv2d(self.reduction_channels, n_classes, kernel_size=3, padding=1)
        else:
            self.y1 = nn.Identity()

    def _combine_prithvi(self, x_unet, x_prithvi):
        """Fuse prithvi features into bottleneck via concat + projection."""
        if x_prithvi.shape[2:] != x_unet.shape[2:]:
            x_prithvi = F.interpolate(x_prithvi, size=x_unet.shape[2:],
                                      mode='bilinear', align_corners=False)
        fused = torch.cat([x_unet, x_prithvi], dim=1)
        return self.bottleneck_proj(fused)

    def encode(self, inputs):
        """Run encoder blocks. Returns list [e1, e2, e3, e4, e5 (bottleneck)]."""
        e1, p1 = self.e1(inputs)
        e2, p2 = self.e2(p1)
        e3, p3 = self.e3(p2)
        e4, p4 = self.e4(p3)
        e5 = self.e5(p4)
        e5 = self.bottleneck_dropout(e5)
        return [e1, e2, e3, e4, e5]

    def decode(self, skips, prithvi_features=None):
        """Run UNet3+ decoder given encoder skip connections.

        Args:
            skips: list from encode() — [e1, e2, e3, e4, e5]
            prithvi_features: optional tensor to fuse at bottleneck
        """
        e1, e2, e3, e4, e5 = skips

        if prithvi_features is not None:
            e5 = self._combine_prithvi(e5, prithvi_features)

        # Decoder level d4
        e1_d4 = F.max_pool2d(e1, kernel_size=8, stride=8)
        e1_d4 = self.e1_d4(e1_d4)

        e2_d4 = F.max_pool2d(e2, kernel_size=4, stride=4)
        e2_d4 = self.e2_d4(e2_d4)

        e3_d4 = F.max_pool2d(e3, kernel_size=2, stride=2)
        e3_d4 = self.e3_d4(e3_d4)

        e4_d4 = self.e4_d4(e4)

        e5_d4 = F.interpolate(e5, scale_factor=2, mode="bilinear", align_corners=True)
        e5_d4 = self.e5_d4(e5_d4)

        d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], dim=1)
        d4 = self.d4(d4)

        # Decoder level d3
        e1_d3 = F.max_pool2d(e1, kernel_size=4, stride=4)
        e1_d3 = self.e1_d3(e1_d3)

        e2_d3 = F.max_pool2d(e2, kernel_size=2, stride=2)
        e2_d3 = self.e2_d3(e2_d3)

        e3_d3 = self.e3_d3(e3)

        e4_d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        e4_d3 = self.e4_d3(e4_d3)

        e5_d3 = F.interpolate(e5, scale_factor=4, mode="bilinear", align_corners=True)
        e5_d3 = self.e5_d3(e5_d3)

        d3 = torch.cat([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3], dim=1)
        d3 = self.d3(d3)

        # Decoder level d2
        e1_d2 = F.max_pool2d(e1, kernel_size=2, stride=2)
        e1_d2 = self.e1_d2(e1_d2)

        e2_d2 = self.e2_d2(e2)

        e3_d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        e3_d2 = self.e3_d2(e3_d2)

        e4_d2 = F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=True)
        e4_d2 = self.e4_d2(e4_d2)

        e5_d2 = F.interpolate(e5, scale_factor=8, mode="bilinear", align_corners=True)
        e5_d2 = self.e5_d2(e5_d2)

        d2 = torch.cat([e1_d2, e2_d2, e3_d2, e4_d2, e5_d2], dim=1)
        d2 = self.d2(d2)

        # Decoder level d1
        e1_d1 = self.e1_d1(e1)

        e2_d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        e2_d1 = self.e2_d1(e2_d1)

        e3_d1 = F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=True)
        e3_d1 = self.e3_d1(e3_d1)

        e4_d1 = F.interpolate(d4, scale_factor=8, mode="bilinear", align_corners=True)
        e4_d1 = self.e4_d1(e4_d1)

        e5_d1 = F.interpolate(e5, scale_factor=16, mode="bilinear", align_corners=True)
        e5_d1 = self.e5_d1(e5_d1)

        d1 = torch.cat([e1_d1, e2_d1, e3_d1, e4_d1, e5_d1], dim=1)
        d1 = self.d1(d1)

        return self.y1(d1)

    def forward(self, inputs, prithvi_features=None):
        skips = self.encode(inputs)
        return self.decode(skips, prithvi_features=prithvi_features)


class RandomHalfDropoutLayer(nn.Module):
    def __init__(self, dropout_prob=None):
        self.dropout_prob = dropout_prob
        super(RandomHalfDropoutLayer, self).__init__()

    def forward(self, x):
        if not self.training or self.dropout_prob == 0:
            return x

        batch_size, num_channels, _, _ = x.size()
        half_channels = num_channels // 2

        strategies = torch.empty(batch_size, 1, 1, 1, device=x.device).uniform_(0, 1)

        mask_prob = self.dropout_prob / 2
        strategies = torch.where(strategies < mask_prob,
                                 torch.tensor(0, device=x.device), strategies)
        strategies = torch.where((strategies >= mask_prob) & (strategies < 2 * mask_prob),
                                 torch.tensor(1, device=x.device), strategies)
        strategies = torch.where(strategies >= 2 * mask_prob,
                                 torch.tensor(2, device=x.device), strategies)

        upper_mask = torch.ones_like(x)
        lower_mask = torch.ones_like(x)

        upper_mask[:, :half_channels, :, :] = 0
        lower_mask[:, half_channels:, :, :] = 0

        x = torch.where(strategies == 0, x * lower_mask * 2, x)
        x = torch.where(strategies == 1, x * upper_mask * 2, x)

        return x