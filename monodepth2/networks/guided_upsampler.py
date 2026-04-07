"""Image-guided learned 2× feature upsampler.

Inspired by FeatUp (arXiv:2403.10516, ICLR 2024):
  "FeatUp: A Model-Agnostic Framework for Features at Any Resolution"

Core idea (Joint Bilateral Upsampling variant):
  Given a low-res feature F (B, C, H, W) and a high-res guide image
  G (B, 3, 2H, 2W), predict per-output-pixel soft-kernels from the
  guide, then apply them to reassemble content-adaptive upsampled
  features.  The kernels are normalised with softmax so the output
  is a convex combination of neighbouring feature values.

This is used in DepthDecoderGuided to replace the nearest-neighbour
`upsample()` calls from the original monodepth2 DepthDecoder.
"""
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedUpsampler(nn.Module):
    """Learned image-guided 2× feature upsampler (FeatUp-inspired).

    Architecture:
      1. Encode the guide image (at target resolution) with a small CNN.
      2. Project the base-upsampled features to the same embedding dim.
      3. Predict k×k soft kernels per output pixel from their concatenation.
      4. Reassemble: weighted sum of unfolded feat_up patches with kernels.

    Args:
        feat_channels: C, number of channels of the feature to upsample.
        guide_channels: channels of the guide image (3 for RGB).
        kernel_size: spatial support of the upsampling kernel (3 or 5).
        embed_dim: internal embedding dimension.
    """

    def __init__(
        self,
        feat_channels: int,
        guide_channels: int = 3,
        kernel_size: int = 3,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        k2 = kernel_size * kernel_size

        # Encode guide at target (2×) resolution
        self.guide_enc = nn.Sequential(
            nn.Conv2d(guide_channels, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Project base-upsampled features to same embed_dim
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_channels, embed_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Predict k² soft-kernel weights per output pixel
        self.kernel_pred = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, k2, 1),
        )

        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                padding=kernel_size // 2)

    def forward(self, feat: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat:  (B, C, H, W)    - low-res feature to upsample.
            guide: (B, 3, 2H, 2W)  - high-res guide image at target res.
        Returns:
            (B, C, 2H, 2W) content-adaptive upsampled features.
        """
        B, C, H, W = feat.shape
        H2, W2 = H * 2, W * 2
        k2 = self.kernel_size ** 2

        # Base upsample to target resolution (will be refined by kernels)
        feat_up = F.interpolate(feat, scale_factor=2, mode="nearest")   # (B, C, 2H, 2W)

        # Embed guide and projected features
        guide_emb = self.guide_enc(guide)          # (B, embed, 2H, 2W)
        feat_emb  = self.feat_proj(feat_up)        # (B, embed, 2H, 2W)

        # Predict and normalise kernels
        combined = torch.cat([guide_emb, feat_emb], dim=1)  # (B, 2*embed, 2H, 2W)
        kernels  = self.kernel_pred(combined)                # (B, k², 2H, 2W)
        kernels  = F.softmax(kernels, dim=1)                 # convex combination

        # Unfold feat_up into local k×k patches for each output pixel
        # unfold output: (B, C*k², 2H*2W)
        feat_patches = self.unfold(feat_up)                              # (B, C*k², N)
        feat_patches = feat_patches.view(B, C, k2, H2 * W2)             # (B, C, k², N)
        kernels      = kernels.view(B, 1, k2, H2 * W2)                  # (B, 1, k², N)

        # Weighted sum over kernel positions -> (B, C, N) -> (B, C, 2H, 2W)
        out = (feat_patches * kernels).sum(dim=2).view(B, C, H2, W2)

        return out
