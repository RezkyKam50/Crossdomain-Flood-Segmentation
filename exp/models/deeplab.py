import torch
import torch.nn as nn


class DeepLabWrapper(nn.Module):
    def __init__(self, model, in_channels=6):
        super().__init__()
        self.model = model

        if in_channels != 3:
            old_conv = model.backbone['conv1']
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True).expand(-1, in_channels - 3, -1, -1)
            model.backbone['conv1'] = new_conv

    def forward(self, sar, optical, elevation, water_occur):
        return self.model(optical)['out']  # optical is already 6 channels