"""A compact 3D U-Net (Çiçek et al., 2016) in plain PyTorch.

Deliberately small and explicit: every layer is visible in ~80 lines, which
makes the model easy to review, test and reason about. The demo configuration
(base_channels=8, depth=3) has 350,827 parameters and trains on a laptop CPU.

    encoder:  [conv block] -> pool -> [conv block] -> pool -> ... -> bottleneck
    decoder:  upsample -> concat skip -> [conv block] -> ... -> 1x1x1 conv -> logits
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig


class ConvBlock(nn.Sequential):
    """Two 3x3x3 convolutions, each followed by instance norm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class UNet3D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int, depth: int) -> None:
        super().__init__()
        widths = [base_channels * 2**level for level in range(depth + 1)]

        self.encoders = nn.ModuleList()
        channels = in_channels
        for width in widths[:-1]:
            self.encoders.append(ConvBlock(channels, width))
            channels = width
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.bottleneck = ConvBlock(widths[-2], widths[-1])

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.upsamplers.append(nn.ConvTranspose3d(width * 2, width, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width))

        self.head = nn.Conv3d(widths[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C_in, Z, Y, X) -> (N, num_classes, Z, Y, X) logits.

        Spatial sizes must be divisible by 2**depth (checked at config load).
        """
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upsample, decoder, skip in zip(
            self.upsamplers, self.decoders, reversed(skips), strict=True
        ):
            x = decoder(torch.cat([upsample(x), skip], dim=1))
        return self.head(x)


def build_model(config: ModelConfig) -> UNet3D:
    return UNet3D(config.in_channels, config.num_classes, config.base_channels, config.depth)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
