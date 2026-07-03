"""Demo: classic interpolation vs NAF feature upsampling on monodepth2 features.

Pipeline
--------
    KITTI image  ->  monodepth2 ResNet encoder  ->  pick one feature map
                 ->  upsample it to full image resolution with
                       (1) classic interpolation (bilinear / bicubic)
                       (2) NAF (Neighborhood Attention Filtering, image-guided)
                 ->  plot [ RGB | classic-upsampled | NAF-upsampled ]  (features shown via PCA->RGB)
                 ->  report VRAM + time for each upsampling method.
"""
import os
import time
import lovely_tensors as lt
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config.conf import Conf
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.models.posenet.resnet_pose_cnn import ResnetEncoder
from src.models.hyenadepth.naf.naf import NAF
from src.utils import readlines

# NAF pretrained "zero-shot" upsampler checkpoint (Valeo.ai).
NAF_CKPT_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"


def load_kitti_image(data_path, filenames, height, width, sample_index, device):
    """Load one image from the KITTI dataloader, return a (1, 3, H, W) tensor in [0, 1]."""
    dataset = KITTIRAWDataset(data_path, filenames, height, width, frame_idxs=[0], num_scales=4, is_train=False, img_ext=".jpg")
    sample = dataset[sample_index % len(dataset)]
    image = sample[("color", 0, 0)].unsqueeze(0).to(device)  # (1, 3, H, W), [0, 1]
    return image


def build_encoder(weights_path, device):
    """monodepth2 ResNet-18 encoder, returning its 5 multi-scale feature maps."""
    encoder = ResnetEncoder(num_layers=18, pretrained=True, num_input_images=1)
    encoder.from_pretrained(weights_path, device=device)  # loads monodepth2 encoder weights
    return encoder.to(device).eval()


def build_naf(device):
    """NAF image-guided feature upsampler."""
    naf = NAF().to(device).eval()
    state = torch.hub.load_state_dict_from_url(NAF_CKPT_URL, map_location=device, progress=True)
    naf.load_state_dict(state)
    return naf


@torch.no_grad()
def benchmark(fn, device, num_runs=20, warmup=5):
    """Run fn() repeatedly; return (output, avg_time_ms, peak_vram_mb)."""
    is_cuda = device.type == "cuda"
    for _ in range(warmup):
        out = fn()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out = fn()
    if is_cuda:
        torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) / num_runs * 1e3
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if is_cuda else float("nan")
    return out, avg_ms, peak_mb


@torch.no_grad()
def features_to_rgb(feature_list):
    """Project a list of (1, C, H, W) feature maps to RGB with a *shared* PCA basis, so colors are directly comparable across the upsampling methods."""
    pixels = [f[0].reshape(f.shape[1], -1).T.float() for f in feature_list]  # each (H*W, C)
    stacked = torch.cat(pixels, dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(stacked - mean, q=3)  # shared 3D basis

    projected = [(p - mean) @ V[:, :3] for p in pixels]
    lo = torch.cat(projected, dim=0).min(dim=0, keepdim=True).values
    hi = torch.cat(projected, dim=0).max(dim=0, keepdim=True).values
    rgbs = []
    for proj, f in zip(projected, feature_list):
        h, w = f.shape[-2:]
        img = ((proj - lo) / (hi - lo).clamp_min(1e-8)).reshape(h, w, 3)
        rgbs.append(img.cpu().numpy())
    return rgbs


def make_plot(rgb_image, classic_rgb, naf_rgb, classic_stats, naf_stats, interp, level, out_path):
    image_np = rgb_image[0].permute(1, 2, 0).cpu().numpy()
    panels = [
        ("Input RGB", image_np, ""),
        (f"{interp} interpolation", classic_rgb, "{:.2f} ms | {:.0f} MB".format(*classic_stats)),
        ("NAF upsampling", naf_rgb, "{:.2f} ms | {:.0f} MB".format(*naf_stats)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Monodepth2 encoder feature (level {level}) upsampled to full resolution", fontsize=14)
    for ax, (title, img, sub) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title + (f"\n{sub}" if sub else ""), fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    lt.monkey_patch()

    conf = Conf().conf
    data_path = conf["data_path"]
    splits_dir = data_path.replace('kitti_data', 'kitti_splits')
    filenames = readlines(os.path.join(splits_dir, conf['evaluation_split'], "test_files.txt"))
    encoder_weights = conf["monodepth2"]["encoder_weights_path"]
    sample = 0 # index into the split file
    level = 2 # choices=[0, 1, 2, 3, 4]; encoder feature level to upsample (0=1/2 ... 4=1/32 resolution)
    interp = "bilinear" # choices=["bilinear", "bicubic"]; classic interpolation baseline
    num_runs = 20
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    H, W = conf["im_sz"]  # [192, 640]

    # 1) KITTI image -> monodepth2 encoder -> feature maps
    image = load_kitti_image(data_path, filenames, H, W, sample, device)
    encoder = build_encoder(encoder_weights, device)
    with torch.no_grad():
        features = encoder(image) # 5 maps at [1/2, 1/4, 1/8, 1/16, 1/32]
    feat = features[level]
    print(f"input image: {tuple(image.shape)} | feature level {level}: {tuple(feat.shape)} -> upsample to ({H}, {W})")

    # 2) classic interpolation
    classic_up, classic_ms, classic_mb = benchmark(lambda: F.interpolate(feat, size=(H, W), mode=interp, align_corners=False if interp != "nearest" else None), device, num_runs=num_runs)

    # 3) NAF upsampling (image-guided)
    naf = build_naf(device=device)
    naf_up, naf_ms, naf_mb = benchmark(lambda: naf(image=image, features=feat, output_size=(H, W)), device, num_runs=num_runs)

    # 4) report
    print("\n{:<22}{:>14}{:>16}".format("method", "time (ms)", "peak VRAM (MB)"))
    print("-" * 52)
    print("{:<22}{:>14.3f}{:>16.1f}".format(f"{interp}", classic_ms, classic_mb))
    print("{:<22}{:>14.3f}{:>16.1f}".format("NAF", naf_ms, naf_mb))

    # 5) plots (features -> RGB via a shared PCA basis)
    classic_rgb, naf_rgb = features_to_rgb([classic_up, naf_up])
    comparison_plot = "assets/naf_demo_upsampling_comparison.png"
    make_plot(image, classic_rgb, naf_rgb, (classic_ms, classic_mb), (naf_ms, naf_mb), interp, level, comparison_plot)
    print(f"\nsaved: {comparison_plot}")

    # 6) also plot the raw selected-level feature map on its own (low-res, PCA->RGB)
    (feat_rgb,) = features_to_rgb([feat])
    low_res_plot = "assets/naf_demo_low_res_features.png"
    plt.figure(figsize=(8, 8 * feat_rgb.shape[0] / feat_rgb.shape[1]))
    plt.imshow(feat_rgb)
    plt.title(f"Monodepth2 encoder feature level {level} ({feat.shape[1]}ch, {feat.shape[2]}x{feat.shape[3]})")
    plt.axis("off")
    plt.savefig(low_res_plot, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"saved: {low_res_plot}")

