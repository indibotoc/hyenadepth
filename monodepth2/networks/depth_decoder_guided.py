"""DepthDecoder with image-guided upsampling (FeatUp-inspired).

Drop-in replacement for the original DepthDecoder that swaps the
nearest-neighbour `upsample()` calls with GuidedUpsampler modules,
which use the input RGB image as a high-frequency guide to produce
sharper, content-adaptive feature upsampling.

Interface change vs. original:
  forward(input_features)             -> original
  forward(input_features, guide_img)  -> this class
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers import ConvBlock, Conv3x3
from networks.guided_upsampler import GuidedUpsampler


class DepthDecoderGuided(nn.Module):
    """U-Net depth decoder with image-guided learned upsampling.

    Args:
        num_ch_enc: channel counts of the 5 encoder feature maps
                    (e.g. [64, 64, 128, 320, 512] for HyenaPixel).
        scales: which output scales to produce disparity maps at.
        num_output_channels: output channels (1 for depth/disparity).
        use_skips: use skip connections from encoder.
        kernel_size: spatial support of the upsampling kernel (3 or 5).
    """

    def __init__(
        self,
        num_ch_enc,
        scales=range(4),
        num_output_channels=1,
        use_skips=True,
        kernel_size=3,
    ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # --- Decoder convolutions (same as original DepthDecoder) ---
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in  = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s],
                                                   self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))

        # --- One GuidedUpsampler per decoder level ---
        # Level i upsamples features with num_ch_dec[i] channels.
        self.guided_ups = nn.ModuleList([
            GuidedUpsampler(feat_channels=int(self.num_ch_dec[i]),
                            guide_channels=3,
                            kernel_size=kernel_size)
            for i in range(5)
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, guide_img):
        """
        Args:
            input_features: list of 5 encoder feature maps (lowest→highest res).
            guide_img: (B, 3, H, W) full-resolution input image in [0,1].
        Returns:
            dict {("disp", s): tensor} for s in self.scales.
        """
        self.outputs = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)

            # Image-guided 2× upsampling at coarse scales (i>=1).
            # At i=0 (full resolution) use nearest to avoid OOM on large feature maps.
            target_h = x.shape[2] * 2
            target_w = x.shape[3] * 2
            if i >= 1:
                guide_i = F.interpolate(guide_img,
                                        size=(target_h, target_w),
                                        mode="bilinear",
                                        align_corners=False)
                x = self.guided_ups[i](x, guide_i)
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")

            if self.use_skips and i > 0:
                x = torch.cat([x, input_features[i - 1]], dim=1)

            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(
                    self.convs[("dispconv", i)](x))

        return self.outputs
