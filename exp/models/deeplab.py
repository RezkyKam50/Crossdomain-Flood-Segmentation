import torch
import torch.nn as nn


class DeepLabWrapper(nn.Module):
    def __init__(self, model, in_channels=6):
        super().__init__()
        self.model = model

        if in_channels != 3:
            # ResNet50 uses 'conv1', MobileNetV3 uses '0'
            conv_key = 'conv1' if 'conv1' in model.backbone else '0'
            old_conv = model.backbone[conv_key]

            # MobileNetV3's first layer is a Conv2dNormActivation Sequential —
            # the actual Conv2d is at index [0]
            is_sequential = isinstance(old_conv, nn.Sequential)
            actual_conv = old_conv[0] if is_sequential else old_conv

            new_conv = nn.Conv2d(
                in_channels,
                actual_conv.out_channels,
                kernel_size=actual_conv.kernel_size,
                stride=actual_conv.stride,
                padding=actual_conv.padding,
                bias=actual_conv.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = actual_conv.weight
                new_conv.weight[:, 3:, :, :] = actual_conv.weight.mean(dim=1, keepdim=True).expand(
                    -1, in_channels - 3, -1, -1
                )

            if is_sequential:
                # Swap only the Conv2d inside the Sequential, preserving BN + activation
                old_conv[0] = new_conv
            else:
                model.backbone[conv_key] = new_conv

    def forward(self, sar, optical, elevation, water_occur):
        return self.model(optical)['out']