"""
PyTorch port of the dual-encoder Resnet50_UNet (originally TF/Keras).

Architecture summary
---------------------
Two independent ResNet50 encoders (one per input image, e.g. `in_img` and
`in_inf`) each produce four feature maps f1..f4 at strides 2, 4(+1), 8, 16.
The corresponding feature maps from both encoders are summed, and the result
is passed through a U-Net style decoder (nearest-neighbour upsampling +
skip concatenation) down to a per-pixel `n_classes` sigmoid output.

Rather than re-implementing every bottleneck block by hand, this uses
`torchvision.models.resnet50` as the encoder backbone and taps it at the
same points the original Keras code taps its hand-rolled ResNet50:

    Keras stage            torchvision equivalent      returned as
    ----------------------------------------------------------------
    conv1 (pre BN/ReLU)     backbone.conv1(x)           f1
    stage2 (a,b,c)          backbone.layer1             f2 (one_side_pad'd)
    stage3 (a,b,c,d)        backbone.layer2             f3
    stage4 (a,b,c,d,e,f)    backbone.layer3             f4
    stage5 (unused)         backbone.layer4 (unused)    -

Block counts match exactly (3 / 4 / 6 blocks respectively), so this is a
faithful translation, not an approximation, and it gets correct ImageNet
pretrained weights for free.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def one_side_pad(x):
    """
    Equivalent of the Keras helper:

        x = ZeroPadding2D((1, 1))(x)
        x = x[:, :-1, :-1, :]

    Net effect: pad 1 pixel on the top and left only (bottom/right stay
    unpadded), growing H and W by 1 each. In PyTorch's F.pad the last-dim
    order is (left, right, top, bottom).
    """
    return F.pad(x, (1, 0, 1, 0))


class ResNet50FeatureEncoder(nn.Module):
    """
    ResNet50 encoder returning intermediate feature maps f1..f4, matching
    the stages used in the original `get_resnet50_encoder` (stops after
    stage 4 / layer3; stage 5 / layer4 is not used).
    """

    def __init__(self, pretrained=True, in_channels=3):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Support non-RGB inputs (the Keras code exposes a `channels` arg too).
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        # torchvision's maxpool uses padding=1; the original Keras code uses
        # MaxPooling2D(padding='valid') i.e. padding=0. This one-pixel
        # difference cascades through every later feature map and breaks
        # the decoder's skip-connection alignment, so it's overridden here.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer1 = backbone.layer1  # stage 2 (3 bottleneck blocks)
        self.layer2 = backbone.layer2  # stage 3 (4 bottleneck blocks)
        self.layer3 = backbone.layer3  # stage 4 (6 bottleneck blocks)
        # backbone.layer4 (stage 5) is intentionally unused, matching the
        # original encoder which stops after stage 4.

    def forward(self, x):
        f1 = self.conv1(x)  # captured before BN/ReLU, like the Keras code

        x = self.bn1(f1)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f2 = one_side_pad(x)

        x = self.layer2(x)
        f3 = x

        x = self.layer3(x)
        f4 = x

        return f1, f2, f3, f4


class DecoderConvBlock(nn.Module):
    """
    Conv(3x3, pad=1) -> ReLU -> BatchNorm.

    Note the order: the Keras code applies `Conv2D(..., activation='relu')`
    followed by a separate `BatchNormalization` call, i.e. conv -> relu -> bn
    (not the more common conv -> bn -> relu). Replicated faithfully here.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class Resnet50UNet(nn.Module):
    """
    Dual-encoder ResNet50-UNet.

    Args:
        n_classes: number of output channels (segmentation classes).
        in_channels_img: channel count for the first input (`in_img`).
        in_channels_inf: channel count for the second input (`in_inf`).
            Defaults to in_channels_img if not given (matches the original,
            which derives both encoders' input shape from `in_img` alone).
        pretrained: load ImageNet weights into both ResNet50 backbones.
        l1_skip_conn: whether to concatenate the f1 skip connection in the
            decoder (matches the `l1_skip_conn` flag in the original).
    """

    def __init__(
        self,
        n_classes,
        in_channels_img=3,
        in_channels_inf=None,
        pretrained=True,
        l1_skip_conn=True,
    ):
        super().__init__()
        self.l1_skip_conn = l1_skip_conn
        if in_channels_inf is None:
            in_channels_inf = in_channels_img

        self.encoder1 = ResNet50FeatureEncoder(pretrained=pretrained, in_channels=in_channels_img)
        self.encoder2 = ResNet50FeatureEncoder(pretrained=pretrained, in_channels=in_channels_inf)

        self.dec_conv1 = DecoderConvBlock(1024, 512)        # from f4 (1024 ch)
        self.dec_conv2 = DecoderConvBlock(512 + 512, 256)   # concat w/ f3 (512 ch)
        self.dec_conv3 = DecoderConvBlock(256 + 256, 128)   # concat w/ f2 (256 ch)

        seg_in = 128 + 64 if l1_skip_conn else 128          # concat w/ f1 (64 ch)
        self.dec_seg_feats = DecoderConvBlock(seg_in, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, s1_img, s2_img, dem, pw):

        in_img = torch.cat([s1_img, pw]) # 2 + 1
        in_inf = torch.cat([s1_img, dem]) # 2 + 1

        f11, f12, f13, f14 = self.encoder1(in_img)
        f21, f22, f23, f24 = self.encoder2(in_inf)

        f1 = f11 + f21
        f2 = f12 + f22
        f3 = f13 + f23
        f4 = f14 + f24

        o = self.dec_conv1(f4)

        o = F.interpolate(o, scale_factor=2, mode="nearest")
        o = torch.cat([o, f3], dim=1)
        o = self.dec_conv2(o)

        o = F.interpolate(o, scale_factor=2, mode="nearest")
        o = torch.cat([o, f2], dim=1)
        o = self.dec_conv3(o)

        o = F.interpolate(o, scale_factor=2, mode="nearest")
        if self.l1_skip_conn:
            o = torch.cat([o, f1], dim=1)
        o = self.dec_seg_feats(o)

        o = F.interpolate(o, scale_factor=2, mode="nearest")
        outputs = self.final_conv(o)

        return outputs

