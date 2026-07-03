from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from src.models.monodepth2.depth_decoder import ConvBlock, Conv3x3
from src.models.hyenadepth.naf.naf import NAF  


class NAFDepthDecoder(nn.Module):
    """Monodepth2 depth decoder with NAF (Neighborhood Attention Filtering) feature upsampling.

    Identical to Monodepth2 depth decoder except that each decoder stage's 2x feature upsample — originally nearest-neighbour — is replaced by an image-guided NAF upsample. 
    NAF builds a neighbourhood cross-attention between embeddings of the high-res RGB image (queries) and the low-res feature positions (keys), aggregating the feature values, so upsampled features (and the predicted disparity boundaries at every scale) follow image edges instead of blurring across them.

    A single NAF module is shared across all stages (its image encoder is re-applied at each stage's target resolution); 
    
    The decoder takes as input the encoder features plus the full-resolution guidance image.
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, naf_kwargs=None):
        super(NAFDepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder (same conv structure as Monodepth2)
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))

        # single image-guided NAF upsampler shared across the 5 decoder stages.
        naf_kwargs = dict(naf_kwargs or dict())
        self.naf = NAF(**naf_kwargs)

        self.sigmoid = nn.Sigmoid()

    def _naf_up(self, x, guidance):
        h, w = x.shape[-2:]
        return self.naf(image=guidance, features=x, output_size=(h * 2, w * 2))  # NAF 2x upsample (image-guided)

    def forward(self, input_features, guidance):
        self.outputs = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self._naf_up(x, guidance)
            x = [x]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

    def from_pretrained(self, weights_path, device='cpu'):
        loaded_dict_dec = torch.load(weights_path, map_location=device)
        filtered_dict_dec = {k: v for k, v in loaded_dict_dec.items() if k in self.state_dict()}
        self.load_state_dict(filtered_dict_dec)
        self.eval()
