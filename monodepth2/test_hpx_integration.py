"""Quick sanity check: forward pass through HyenaPixelEncoder + decoders."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import networks

def test_encoder_shapes():
    print("=== HyenaPixelEncoder (pretrained=False for speed) ===")
    enc = networks.HyenaPixelEncoder(pretrained=False).cuda()
    x = torch.randn(2, 3, 192, 640).cuda()
    feats = enc(x)
    print(f"num_ch_enc: {enc.num_ch_enc}")
    for i, f in enumerate(feats):
        print(f"  features[{i}]: {tuple(f.shape)}")
    assert enc.num_ch_enc.tolist() == [64, 64, 128, 320, 512]
    assert feats[0].shape == (2,  64,  96, 320)
    assert feats[1].shape == (2,  64,  48, 160)
    assert feats[2].shape == (2, 128,  24,  80)
    assert feats[3].shape == (2, 320,  12,  40)
    assert feats[4].shape == (2, 512,   6,  20)
    print("  PASSED\n")
    return enc, feats

def test_depth_decoder(enc, feats):
    print("=== DepthDecoder (standard) ===")
    dec = networks.DepthDecoder(enc.num_ch_enc, scales=range(4)).cuda()
    out = dec(feats)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    assert ("disp", 0) in out
    print("  PASSED\n")

def test_guided_upsampler():
    print("=== GuidedUpsampler ===")
    from networks.guided_upsampler import GuidedUpsampler
    up = GuidedUpsampler(feat_channels=128).cuda()
    feat  = torch.randn(2, 128, 24, 80).cuda()
    guide = torch.randn(2,   3, 48, 160).cuda()
    out = up(feat, guide)
    assert out.shape == (2, 128, 48, 160), f"Expected (2,128,48,160), got {out.shape}"
    print(f"  {tuple(feat.shape)} -> {tuple(out.shape)}  PASSED\n")

def test_depth_decoder_guided(enc, feats):
    print("=== DepthDecoderGuided (FeatUp-inspired) ===")
    dec = networks.DepthDecoderGuided(enc.num_ch_enc, scales=range(4)).cuda()
    guide = torch.randn(2, 3, 192, 640).cuda()
    out = dec(feats, guide)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    assert ("disp", 0) in out
    print("  PASSED\n")

if __name__ == "__main__":
    enc, feats = test_encoder_shapes()
    test_depth_decoder(enc, feats)
    test_guided_upsampler()
    test_depth_decoder_guided(enc, feats)
    print("All tests passed!")
