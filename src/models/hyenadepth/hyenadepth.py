import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from pytorch_model_summary import summary as psummary
import lovely_tensors as lt

from src.models.hyenadepth.hyena_depth_encoder import HyenaEncoder
from src.models.monodepth2.depth_decoder import DepthDecoder
from src.models.hyenadepth.naf_depth_decoder import NAFDepthDecoder
from src.models.hyenadepth.naf.naf import NAF


class HyenaDepth(nn.Module):
    """Monodepth2-style depth network with a Hyena-based backbone.
    Drop-in replacement for Monodepth2: the ResNet encoder is swapped for a Hyena-based backbone while the Monodepth2 depth decoder is reused unchanged.

    Two upsampling sites can each be set to an image-guided learned upsampler (NAF) instead of classic upsampling (nearest-neighbour/bilinear/bicubic):

    - ``decoder_upsample`` in {"nearest", "naf"}: how the decoder upsamples its feature maps by 2x at every stage. 
    - ``disp_upsample`` in {"bilinear", "bicubic", "naf"}: how the trainer upsamples each scale's predicted disparity to full resolution for the reprojection loss.
    """

    def __init__(self, pretrained=True, scales=range(4), decoder_upsample="nearest", disp_upsample="bicubic", naf_kwargs=None):
        super(HyenaDepth, self).__init__()
        self.decoder_upsample = decoder_upsample
        self.disp_upsample = disp_upsample
        naf_kwargs = dict(naf_kwargs or dict())

        self.encoder = HyenaEncoder(pretrained=pretrained)

        # ---- decoder (feature upsampling) ----
        num_ch_enc = self.encoder.num_ch_enc
        if decoder_upsample == "naf":
            self.decoder = NAFDepthDecoder(num_ch_enc=num_ch_enc, scales=scales, naf_kwargs=naf_kwargs)
        else:  # "nearest" -> original Monodepth2 decoder
            self.decoder = DepthDecoder(num_ch_enc=num_ch_enc, scales=scales)

        # ---- disparity -> full-res upsampler (used by the trainer) ----
        # NAF is image-guided and run on the 1-channel disparity. NAF cross-attends the image embedding over the low-res disparity (single attention head, since the value has one channel).
        if disp_upsample == "naf":
            disp_naf_kwargs = dict(naf_kwargs)
            disp_naf_kwargs["heads_attn"] = 1  # value (disparity) has a single channel
            self.disp_upsampler = NAF(**disp_naf_kwargs)

    def forward(self, x):
        features = self.encoder(x)
        if self.decoder_upsample == "naf":
            disparity = self.decoder(features, x)  # x is the high-res RGB guidance
        else:
            disparity = self.decoder(features)
        return disparity

    def _disp_naf(self, disp8, guidance):
        return self.disp_upsampler(image=guidance, features=disp8, output_size=guidance.shape[-2:])

    def upsample_disp(self, disp, guidance):
        """Image-guided upsampling of a disparity map to the guidance resolution.

        disp: (B, 1, h, w) at some scale; guidance: (B, 3, H, W) full-res RGB image.
        Returns disp upsampled to (B, 1, H, W) with edges guided by the image.
        """
        # NAF's CUTLASS-FNA backend needs the value head dim to be a multiple of 8. Disparity has 1 channel, so we replicate it to 8 channels (a single attention head -> head dim 8). 
        # The attention weights come from the image embedding and are value-independent, so every output channel is the same upsampled disparity; we return the first.
        disp8 = disp.repeat(1, 8, 1, 1)
        up = self._disp_naf(disp8, guidance)
        return up[:, :1]

    def from_pretrained(self, encoder_weights_path, decoder_weights_path, weights_path=None, device='cpu'):
        if weights_path is not None:
            # full model saved as a single file
            loaded_dict_dec = torch.load(weights_path, map_location=device)
            filtered_dict_dec = {k: v for k, v in loaded_dict_dec.items() if k in self.state_dict()}
            self.load_state_dict(filtered_dict_dec)
            self.eval()
        else:
            self.encoder.from_pretrained(encoder_weights_path, device)
            self.decoder.from_pretrained(decoder_weights_path, device)


if __name__ == "__main__":

    lt.monkey_patch()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pretrained = True
    scales = [0, 1, 2, 3]
    decoder_upsample = "naf" # "nearest" | "naf"
    disp_upsample = "naf" # "bilinear" | "bicubic" | "naf"  (the trainer applies disp upsampling)
    naf_kwargs = dict(dim=256, heads_attn=2, heads_rope=4, kernel_size=3, img_layers=2)

    model = HyenaDepth(pretrained=pretrained, scales=scales, decoder_upsample=decoder_upsample, disp_upsample=disp_upsample, naf_kwargs=naf_kwargs).to(device)
    input = torch.rand(1, 3, 192, 640).to(device)

    ######## VISUALIZING THE ARCHITECTURE ########
    architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
    # print(model)

    output = model(input)
    for s in scales:
        print("disp", s, ":", output[("disp", s)].shape)
