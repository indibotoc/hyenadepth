import torch
import torch.nn as nn
import torch.nn.functional as F


def _mat_to_axis_angle(R):
    """
    Rodrigues formula: rotation matrix (..., 3, 3) -> axis-angle (..., 3).
    Axis-angle vector direction = rotation axis, magnitude = rotation angle.
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # (B,)

    # Skew-symmetric part: 2*sin(θ) * [axis]×
    r = torch.stack([
        R_flat[:, 2, 1] - R_flat[:, 1, 2],
        R_flat[:, 0, 2] - R_flat[:, 2, 0],
        R_flat[:, 1, 0] - R_flat[:, 0, 1],
    ], dim=1)  # (B, 3)

    sin_theta = torch.clamp(torch.sin(theta), min=1e-7).unsqueeze(1)
    axis = r / (2.0 * sin_theta)

    return (theta.unsqueeze(1) * axis).reshape(batch_shape + (3,))


class VGGTPoseNet(nn.Module):
    """
    Wraps VGGT to predict relative camera pose between two input frames.

    VGGT predicts per-frame world-to-camera extrinsics in a shared canonical
    frame. Given inputs [frame_a, frame_b], this module computes the relative
    transformation T_{a→b} (points from cam_a to cam_b) and converts it to the
    axis-angle + translation format expected by monodepth2.

    Interface matches PoseDecoder:
      forward(frames) where frames is a list of 2 tensors (B, 3, H, W) in [0, 1]
      returns axisangle (B, 1, 1, 3), translation (B, 1, 1, 3)

    Requires --pose_model_input pairs (default).
    """

    PATCH_SIZE = 14  # VGGT's ViT patch size
    NATIVE_SIZE = 518  # resolution VGGT was trained at

    def __init__(self, pretrained=True, freeze=True, img_size=None):
        super().__init__()

        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self._pose_enc_to_extri = pose_encoding_to_extri_intri
        self._freeze = freeze

        # Round img_size to nearest multiple of patch_size
        if img_size is None:
            img_size = self.NATIVE_SIZE
        img_size = (img_size // self.PATCH_SIZE) * self.PATCH_SIZE
        self.img_size = img_size

        if pretrained:
            model = VGGT.from_pretrained("facebook/VGGT-1B")
        else:
            model = VGGT(enable_point=False, enable_depth=False, enable_track=False)

        # Drop unused prediction heads to save GPU memory
        model.point_head = None
        model.depth_head = None
        model.track_head = None

        self.vggt = model

        if freeze:
            for p in self.vggt.parameters():
                p.requires_grad_(False)

    def _vggt_forward(self, imgs):
        if self._freeze:
            with torch.no_grad():
                return self.vggt(imgs)
        return self.vggt(imgs)

    def forward(self, frames):
        B, _, H, W = frames[0].shape

        imgs = torch.stack(frames, dim=1)  # (B, 2, 3, H, W)

        if H != self.img_size or W != self.img_size:
            imgs = F.interpolate(
                imgs.view(B * 2, 3, H, W),
                size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            ).view(B, 2, 3, self.img_size, self.img_size)

        preds = self._vggt_forward(imgs)
        pose_enc = preds["pose_enc"]  # (B, 2, 9): [T(3), quat(4), fov_h, fov_w]

        extrinsics, _ = self._pose_enc_to_extri(
            pose_enc,
            image_size_hw=(self.img_size, self.img_size),
            build_intrinsics=False,
        )
        # extrinsics: (B, 2, 3, 4) = [R | t], world-to-camera (OpenCV convention)

        Ra, ta = extrinsics[:, 0, :, :3], extrinsics[:, 0, :, 3]  # (B, 3, 3), (B, 3)
        Rb, tb = extrinsics[:, 1, :, :3], extrinsics[:, 1, :, 3]

        # Relative pose T_{a→b}: P_b = R_rel @ P_a + t_rel
        R_rel = Rb @ Ra.transpose(-1, -2)
        t_rel = tb - (R_rel @ ta.unsqueeze(-1)).squeeze(-1)

        axisangle = _mat_to_axis_angle(R_rel).view(B, 1, 1, 3)
        translation = t_rel.view(B, 1, 1, 3)

        return axisangle, translation
