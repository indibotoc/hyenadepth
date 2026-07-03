import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial

from src.models.hyenadepth.hyena.metaformer import MetaFormer, MlpHead
from src.models.hyenadepth.hyena.hyenapixel import HyenaPixelOperator

_DEFAULT_BACKBONE_WEIGHTS = os.path.join(os.path.dirname(__file__), "hyena", "weights", "hyena_backbone.pth")


class HyenaEncoder(nn.Module):
    """ResNet-style encoder built on a Hyena-based backbone.

    The Monodepth2 depth decoder expects 5 feature maps at [1/2, 1/4, 1/8, 1/16, 1/32].
    The Hyena backbone produces features at [1/4, 1/8, 1/16, 1/32] only.
    To match the decoder contract and mirror Resnet Encoder, we add a lightweight conv stem that supplies the missing 1/2-resolution feature, then append the 4 Hyena stages.

    Args:
        pretrained (bool): if True, loads ImageNet-1k pretrained backbone weights from `backbone_weights`.
        stem_channels (int): channels of the added 1/2-resolution stem feature.
        backbone_weights (str): path to the local backbone checkpoint (used when pretrained=True).
    """
    def __init__(self, pretrained=True, stem_channels=64, backbone_weights=_DEFAULT_BACKBONE_WEIGHTS):
        super(HyenaEncoder, self).__init__()

        # backbone token mixers: one HyenaPixelOperator per stage (use_layernorm=True), with decreasing long-conv kernel sizes as resolution drops.
        token_mixers = [
            partial(HyenaPixelOperator, filter_emb_dim=32, long_kernel_size=111, use_layernorm=True),
            partial(HyenaPixelOperator, filter_emb_dim=32, long_kernel_size=55, use_layernorm=True),
            partial(HyenaPixelOperator, filter_emb_dim=48, long_kernel_size=27, use_layernorm=True),
            partial(HyenaPixelOperator, filter_emb_dim=64, long_kernel_size=13, use_layernorm=True),
        ]

        # Build the backbone in feature-extraction mode: returns one (B, C, H, W) tensor per stage.
        self.backbone = MetaFormer(
            depths=[3, 3, 9, 3],
            dims=[64, 128, 320, 512],
            token_mixers=token_mixers,
            head_fn=MlpHead,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        if pretrained:
            state_dict = torch.load(backbone_weights, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False) # strict=False because the classification head/norm of the pretrained checkpoint are dropped in features_only mode and per-stage feature norms are added (fnorm).

        stage_dims = list(self.backbone.dims)  # e.g. [64, 128, 320, 512]

        # 1/2-resolution stem to provide the high-res skip the decoder expects.
        self.stem_half = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        self.num_ch_enc = np.array([stem_channels] + stage_dims) # [1/2, 1/4, 1/8, 1/16, 1/32]

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.225 # Same input normalization as for the ResnetEncoder, for consistency with the rest of the self-supervised depth pipeline.

        self.features = [self.stem_half(x)]          # 1/2 resolution
        self.features += self.backbone(x)            # 1/4, 1/8, 1/16, 1/32
        return self.features

    def from_pretrained(self, weights_path, device='cpu'):
        loaded_dict = torch.load(weights_path, map_location=device)
        filtered_dict = {k: v for k, v in loaded_dict.items() if k in self.state_dict()}
        self.load_state_dict(filtered_dict)
        self.eval()
