"""HyenaPixel encoder for monodepth2.

Drop-in replacement for ResnetEncoder using hpx_former_s18 (29M params)
pretrained on ImageNet-1k.

Feature map sizes for input (B, 3, H, W):
  features[0]: (B,  64, H/2,  W/2)   - lightweight stem conv
  features[1]: (B,  64, H/4,  W/4)   - HyenaPixel stage 0
  features[2]: (B, 128, H/8,  W/8)   - HyenaPixel stage 1
  features[3]: (B, 320, H/16, W/16)  - HyenaPixel stage 2
  features[4]: (B, 512, H/32, W/32)  - HyenaPixel stage 3

num_ch_enc = [64, 64, 128, 320, 512]  (compatible with DepthDecoder)
"""
from __future__ import absolute_import, division, print_function

import sys
import os

import numpy as np
import torch
import torch.nn as nn

# Make hyenapixel importable (handles both installed and source-tree cases)
_HPX_SRC = os.path.join(os.path.dirname(__file__),
                        "../../../HyenaPixel/src")
if os.path.isdir(_HPX_SRC) and _HPX_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_HPX_SRC))

import hyenapixel.models  # noqa: F401 – registers hpx_former_s18 in timm
import timm

# ImageNet per-channel mean / std (HyenaPixel was trained with these)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class HyenaPixelEncoder(nn.Module):
    """HyenaPixel backbone encoder for self-supervised depth estimation.

    Wraps hpx_former_s18 (features_only mode) and prepends a lightweight
    stride-2 stem so that the output list matches monodepth2's 5-level
    encoder convention expected by DepthDecoder and PoseDecoder.

    Args:
        pretrained: load HuggingFace ImageNet-1k checkpoint (recommended).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 320, 512])

        # --- Lightweight stride-2 stem (not in pretrained checkpoint) ---
        # Provides features[0] at H/2 × W/2, bridging full-res and stage-0.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # --- HyenaPixel backbone (stages 0-3, strides 4/8/16/32) ---
        self.backbone = timm.create_model(
            "hpx_former_s18",
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3],
            strict=False,          # pretrained ckpt has head/norm not in features_only model
        )

        # --- ImageNet normalisation buffers ---
        # monodepth2 feeds images in [0,1]; we normalise here before backbone.
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) float tensor in [0, 1] range.
        Returns:
            List of 5 feature tensors at strides [2, 4, 8, 16, 32].
        """
        x_norm = (x - self.mean) / self.std   # ImageNet normalisation

        f0 = self.stem(x_norm)                # (B,  64, H/2,  W/2)
        hpx = self.backbone(x_norm)           # list of 4 [B, C, H/s, W/s]
        # hpx[0]: (B,  64, H/4,  W/4)
        # hpx[1]: (B, 128, H/8,  W/8)
        # hpx[2]: (B, 320, H/16, W/16)
        # hpx[3]: (B, 512, H/32, W/32)

        return [f0] + hpx
