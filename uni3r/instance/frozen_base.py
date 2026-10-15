"""A-owned, full-forward frozen Uni3R feature boundary (2-view fp32 v0).

No head removal, GT input, Gaussian filtering, or change to the base forward.
The model must have been loaded from the checkpoint recorded in provenance.
"""
from copy import deepcopy

import torch
from torch import nn

from .contracts import (CONTRACT_VERSION, GAUSSIAN_FLATTEN_ORDER, TOKEN_LAYERS,
                        validate_frozen_scene_batch)


def tensor_items(value, prefix=""):
    """Stable tensor paths, including nested Gaussian payloads."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key != "metadata":
                yield from tensor_items(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from tensor_items(item, f"{prefix}[{i}]")


def map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [map_tensors(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(v, fn) for v in value)
    return deepcopy(value)


class FrozenUni3R(nn.Module):
    def __init__(self, base: nn.Module, provenance: dict):
        super().__init__()
        self.base = base
        self.provenance = deepcopy(provenance)
        self.base.requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def forward(self, images, intrinsics, sample_metadata):
        if images.ndim != 5 or tuple(images.shape[1:]) != (2, 3, 256, 256):
            raise ValueError("A v0 requires [B,2,3,256,256] images")
        if intrinsics.shape != (images.shape[0], 2, 3, 3):
            raise ValueError("Expected pixel intrinsics [B,2,3,3]")
        if images.dtype != torch.float32 or intrinsics.dtype != torch.float32:
            raise ValueError("A v0 requires fp32 inputs")
        if images.device != intrinsics.device or not torch.isfinite(intrinsics).all():
            raise ValueError("Invalid intrinsics or device mismatch")
        if not torch.isfinite(images).all() or images.min() < -1 or images.max() > 1:
            raise ValueError("Expected finite RGB in [-1,1]")
        if torch.is_autocast_enabled() or any(p.requires_grad for p in self.base.parameters()):
            raise RuntimeError("Base must remain frozen and fp32 autocast disabled")
        self.base.eval()
        captured = {}

        def capture(_module, _inputs, output):
            tokens = output["tokens"]
            start = int(tokens["patch_start_idx"])
            # VGGT_IntrEmb.forward has already removed the trailing intrinsic token.
            if start != 5 or len(tokens["aggregated_tokens_list"]) != 24:
                raise ValueError("Unexpected Uni3R special-token/layer layout")
            captured["patch_tokens"] = {}
            for layer in TOKEN_LAYERS:
                t = tokens["aggregated_tokens_list"][layer][:, :, start:]
                if tuple(t.shape) != (images.shape[0], 2, 256, 2048):
                    raise ValueError(f"Unexpected patch-only shape at layer {layer}: {t.shape}")
                captured["patch_tokens"][layer] = t.detach().clone()
            captured["means"] = output["world_points"].detach().clone()
            captured["geometry_confidence"] = output["world_points_conf"].detach().clone()

        hook = self.base.vggt.register_forward_hook(capture)
        try:
            views = [{"img": images[:, i], "camera_intrinsics": intrinsics[:, i]} for i in range(2)]
            predictions = self.base(views)
        finally:
            hook.remove()
        if not captured:
            raise RuntimeError("VGGT feature hook did not run")
        raw = map_tensors(predictions, lambda t: t.detach())
        if not torch.equal(torch.stack([v["means"] for v in raw], 1), captured["means"]):
            raise RuntimeError("Gaussian means differ from exported predicted XYZ")
        batch = dict(images=images.detach().clone(), intrinsics=intrinsics.detach().clone(),
                     **captured, gaussians={"views": raw, "source_indices": [
                         torch.arange(2*256*256, device=images.device, dtype=torch.int64)
                         for _ in range(images.shape[0])]})
        meta = deepcopy(sample_metadata)
        meta.update(deepcopy(self.provenance))
        meta.update(contract_version=CONTRACT_VERSION, evidence_kind="real", input_size=[256, 256],
                    rgb_normalization="minus_one_to_one", token_layers=list(TOKEN_LAYERS),
                    patch_token_order="row_major_yx", special_tokens_removed=True,
                    special_token_removal="VGGT_IntrEmb removes the final intrinsic token; A slices off 1 camera + 4 register tokens (patch_start_idx=5).",
                    intrinsics_convention="pixel", means_source="uni3r_prediction",
                    coordinate_system="Unmodified Uni3R point-head shared prediction frame, native model units; no metric calibration, GT pose, or GT alignment. views[i].extr is predicted world-to-camera.",
                    geometry_confidence_semantics="VGGT point_head expp1: 1 + exp(raw score), an uncalibrated nonnegative score, not a probability.",
                    gaussian_flatten_order=GAUSSIAN_FLATTEN_ORDER, gaussian_filtered=False,
                    gaussian_schema="A.raw_views.v1: views is the original 2-element Uni3R output list, each field retains batch dimension. Gaussian attributes [B,H,W,...]; source_indices[b]=arange(V*H*W), view-major then row-major. Extra camera/depth/distiller fields retain original shapes. rotations are raw model values; covariance builder interprets xyzw, legacy gsplat path passes raw rotations unchanged.")
        meta["tensor_dtypes"] = {k: str(t.dtype) for k, t in tensor_items(batch)}
        batch["metadata"] = meta
        validate_frozen_scene_batch(batch)
        return batch
