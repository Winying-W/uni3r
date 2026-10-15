"""Executable B proposal, not an A/C-approved or real-model-verified interface.

Default profile: two 256x256 views, patch size 16, layers 4/11/17/23,
2048 token channels. A smaller explicit ContractSpec is for synthetic tests.
All validators are read-only and raise ContractError; they never normalize,
detach, clamp, renumber, repair or transfer an input tensor.

IDs: -1 unknown, 0 trusted background/stuff, positive scene-wide objects.
Unknown/invalid pixels have zero confidence. Valid GT weights normally equal
one (the producer supplies them); explicit weights in [0,1] are supported.
Matching must use the scene-wide union, never independent per-view matches.
Physical object identity and checkpoint provenance require producer evidence;
they cannot be established from shapes or labels alone.

Gaussian payload fields are opaque until A confirms its exported schema.
Only `source_indices` is reserved here: after filtering it is a sequence of B
int64 vectors, mapping each retained Gaussian back to view*H*W+y*W+x.
Payload/index correspondence remains A's schema-specific responsibility.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any, Literal, TypedDict

import torch
from torch import Tensor

CONTRACT_VERSION = "0.1.0"
CONTRACT_STATUS = "PROPOSED_FOR_TEAM_INTEGRATION"
TOKEN_LAYERS = (4, 11, 17, 23)  # zero-based indices in the 24-entry list
GAUSSIAN_FLATTEN_ORDER = "index = view * H * W + y * W + x"
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


class FrozenSceneBatch(TypedDict):
    images: Tensor
    intrinsics: Tensor
    patch_tokens: dict[int, Tensor]
    means: Tensor
    geometry_confidence: Tensor
    gaussians: Mapping[str, Any]
    metadata: Mapping[str, Any]


class InstanceTargets(TypedDict):
    instance_ids: Tensor
    valid_mask: Tensor
    confidence: Tensor
    scene_id: Sequence[str]
    frame_ids: Sequence[Sequence[str]]
    label_source: Literal["gt", "pseudo"]


class InstancePrediction(TypedDict):
    mask_logits: Tensor
    objectness_logits: Tensor
    query_features: Tensor


@dataclass(frozen=True)
class ContractSpec:
    height: int = 256
    width: int = 256
    views: int = 2
    patch_size: int = 16
    token_dim: int = 2048
    query_dim: int = 256

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"ContractSpec.{key} must be a positive integer")
        if self.height % self.patch_size or self.width % self.patch_size:
            raise ValueError("ContractSpec image size must be divisible by patch_size")


DEFAULT_SPEC = ContractSpec()


class ContractError(ValueError):
    """An input violates the contract; the original input is unchanged."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _fields(value: Any, keys: Sequence[str], path: str) -> Mapping:
    _require(isinstance(value, Mapping), f"{path}: expected a mapping")
    for key in keys:
        _require(key in value, f"{path}.{key}: missing required field")
    return value


def _text(value: Any, path: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonempty string")


def _sequence(value: Any, length: int, path: str) -> None:
    _require(isinstance(value, (list, tuple)) and len(value) == length,
             f"{path}: expected list/tuple of length {length}")


def _tensor(value: Any, path: str, shape: tuple | None = None,
            dtype: torch.dtype | None = None, frozen: bool = False,
            device: torch.device | None = None) -> Tensor:
    _require(isinstance(value, Tensor), f"{path}: expected torch.Tensor")
    _require(value.layout == torch.strided and value.device.type != "meta",
             f"{path}: expected materialized dense tensor")
    if shape is not None:
        _require(tuple(value.shape) == shape, f"{path}: expected shape {shape}, got {tuple(value.shape)}")
    if dtype is not None:
        _require(value.dtype == dtype, f"{path}: expected dtype {dtype}, got {value.dtype}")
    else:
        _require(value.dtype in FLOAT_DTYPES, f"{path}: expected floating dtype, got {value.dtype}")
    _require(bool(torch.isfinite(value).all()), f"{path}: NaN/Inf are forbidden, including ignored pixels")
    if frozen:
        _require(not value.requires_grad, f"{path}: frozen/target tensors must not require gradients")
    if device is not None:
        _require(value.device == device, f"{path}: device differs from other tensors")
    return value


def _identifiers(scene_ids: Any, frame_ids: Any, b: int, v: int, path: str) -> None:
    _sequence(scene_ids, b, f"{path}.scene_id")
    _sequence(frame_ids, b, f"{path}.frame_ids")
    for i in range(b):
        _text(scene_ids[i], f"{path}.scene_id[{i}]")
        _sequence(frame_ids[i], v, f"{path}.frame_ids[{i}]")
        for j, frame in enumerate(frame_ids[i]):
            _text(frame, f"{path}.frame_ids[{i}][{j}]")
        _require(len(set(frame_ids[i])) == v, f"{path}.frame_ids[{i}]: duplicate input frames")


def _payload_tensors(value: Any, path: str):
    if isinstance(value, Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require(isinstance(key, str), f"{path}: payload keys must be strings")
            yield from _payload_tensors(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _payload_tensors(item, f"{path}[{i}]")
    else:
        _require(value is None or isinstance(value, (str, bool, int, float)),
                 f"{path}: unsupported opaque payload value")
        if isinstance(value, float):
            import math
            _require(math.isfinite(value), f"{path}: NaN/Inf scalar is forbidden")


def _metadata(meta: Any, tensors: Mapping[str, Tensor], b: int, spec: ContractSpec) -> None:
    required = ("contract_version", "evidence_kind", "base_checkpoint", "uni3r_git_commit",
                "uni3r_worktree_dirty", "input_size", "rgb_normalization", "token_layers",
                "patch_token_order", "special_tokens_removed", "special_token_removal",
                "scene_id", "frame_ids", "crop_resize", "intrinsics_convention",
                "coordinate_system", "means_source", "geometry_confidence_semantics",
                "tensor_dtypes", "gaussian_flatten_order", "gaussian_filtered", "gaussian_schema")
    m = _fields(meta, required, "metadata")
    _require(m["contract_version"] == CONTRACT_VERSION, "metadata.contract_version: incompatible version")
    _require(m["evidence_kind"] in ("real", "synthetic"), "metadata.evidence_kind: expected real or synthetic")
    cp = _fields(m["base_checkpoint"], ("name", "sha256"), "metadata.base_checkpoint")
    _require(type(m["uni3r_worktree_dirty"]) is bool, "metadata.uni3r_worktree_dirty: expected bool")
    if m["evidence_kind"] == "real":
        _require(spec == DEFAULT_SPEC, "metadata: real v0.1.0 caches require the default 2-view profile")
        _require(cp["name"] == "2views.pth", "metadata.base_checkpoint.name: phase one requires 2views.pth")
        _require(isinstance(cp["sha256"], str) and re.fullmatch(r"[0-9a-fA-F]{64}", cp["sha256"]) is not None,
                 "metadata.base_checkpoint.sha256: expected SHA256 hex")
        _require(isinstance(m["uni3r_git_commit"], str) and re.fullmatch(r"[0-9a-fA-F]{40}", m["uni3r_git_commit"]) is not None,
                 "metadata.uni3r_git_commit: expected full Git commit")
        if m["uni3r_worktree_dirty"]:
            _fields(m, ("uni3r_source_sha256",), "metadata")
            fingerprint = m["uni3r_source_sha256"]
            _require(isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint) is not None,
                     "metadata.uni3r_source_sha256: dirty source requires a SHA256 fingerprint")
        _require(m["means_source"] == "uni3r_prediction", "metadata.means_source: GT XYZ is forbidden")
    else:
        _require(cp["name"] == "SYNTHETIC_NO_CHECKPOINT" and cp["sha256"] is None,
                 "metadata.base_checkpoint: synthetic inputs must not claim checkpoint provenance")
        _require(m["uni3r_git_commit"] == "SYNTHETIC" and m["means_source"] == "synthetic",
                 "metadata: synthetic inputs must identify synthetic source")
    _sequence(m["input_size"], 2, "metadata.input_size")
    _require(tuple(m["input_size"]) == (spec.height, spec.width), "metadata.input_size: mismatch")
    _sequence(m["token_layers"], 4, "metadata.token_layers")
    _require(all(type(x) is int for x in m["token_layers"]) and tuple(m["token_layers"]) == TOKEN_LAYERS,
             "metadata.token_layers: expected zero-based [4, 11, 17, 23]")
    _require(m["patch_token_order"] == "row_major_yx", "metadata.patch_token_order: expected row_major_yx")
    _require(m["special_tokens_removed"] is True, "metadata.special_tokens_removed: must be True")
    for key in ("special_token_removal", "coordinate_system", "geometry_confidence_semantics", "gaussian_schema"):
        _text(m[key], f"metadata.{key}")
        if m["evidence_kind"] == "real":
            _require(m[key].strip().lower() not in ("unknown", "unconfirmed", "pending", "synthetic"),
                     f"metadata.{key}: unresolved real-cache semantics")
    _require(m["intrinsics_convention"] in ("pixel", "normalized_by_image_size"),
             "metadata.intrinsics_convention: expected pixel or normalized_by_image_size")
    _require(m["rgb_normalization"] in ("minus_one_to_one", "zero_to_one"),
             "metadata.rgb_normalization: unsupported convention")
    low = -1 if m["rgb_normalization"] == "minus_one_to_one" else 0
    images = tensors["images"]
    _require(bool(((images >= low) & (images <= 1)).all()), "images: values violate metadata.rgb_normalization")
    _identifiers(m["scene_id"], m["frame_ids"], b, spec.views, "metadata")
    _sequence(m["crop_resize"], b, "metadata.crop_resize")
    for i, row in enumerate(m["crop_resize"]):
        _sequence(row, spec.views, f"metadata.crop_resize[{i}]")
        for j, item in enumerate(row):
            path = f"metadata.crop_resize[{i}][{j}]"
            item = _fields(item, ("original_size", "crop_xyxy", "resize_hw", "description"), path)
            _text(item["description"], path + ".description")
            _sequence(item["original_size"], 2, path + ".original_size")
            _require(all(type(x) is int and x > 0 for x in item["original_size"]), path + ": invalid original_size")
            _sequence(item["crop_xyxy"], 4, path + ".crop_xyxy")
            _require(all(type(x) is int for x in item["crop_xyxy"]), path + ": crop must use integer pixel edges")
            x0, y0, x1, y1 = item["crop_xyxy"]
            oh, ow = item["original_size"]
            _require(0 <= x0 < x1 <= ow and 0 <= y0 < y1 <= oh, path + ": crop outside original image")
            _sequence(item["resize_hw"], 2, path + ".resize_hw")
            _require(tuple(item["resize_hw"]) == (spec.height, spec.width), path + ": resize mismatch")
    dtype_map = _fields(m["tensor_dtypes"], tuple(tensors), "metadata.tensor_dtypes")
    _require(set(dtype_map) == set(tensors), "metadata.tensor_dtypes: unexpected tensor paths")
    for name, tensor in tensors.items():
        _require(dtype_map[name] == str(tensor.dtype), f"metadata.tensor_dtypes.{name}: dtype mismatch")
    _require(m["gaussian_flatten_order"] == GAUSSIAN_FLATTEN_ORDER, "metadata.gaussian_flatten_order: mismatch")
    _require(type(m["gaussian_filtered"]) is bool, "metadata.gaussian_filtered: expected bool")


def validate_frozen_scene_batch(batch: FrozenSceneBatch, *, spec: ContractSpec = DEFAULT_SPEC) -> None:
    """Validate detached features and provenance declarations, not their truth.

    Geometry confidence is finite and nonnegative, with no upper bound; current
    VGGT uses expp1. Gaussian raw fields are not assigned invented semantics.
    """
    x = _fields(batch, ("images", "intrinsics", "patch_tokens", "means", "geometry_confidence", "gaussians", "metadata"), "batch")
    images = _tensor(x["images"], "images", frozen=True)
    _require(images.ndim == 5 and images.shape[0] > 0, "images: expected nonempty [B,V,3,H,W]")
    b, v, h, w = images.shape[0], spec.views, spec.height, spec.width
    _tensor(images, "images", (b, v, 3, h, w))
    tensors = {"images": images}
    for name, shape in (("intrinsics", (b, v, 3, 3)), ("means", (b, v, h, w, 3)), ("geometry_confidence", (b, v, h, w))):
        tensors[name] = _tensor(x[name], name, shape, frozen=True, device=images.device)
    intr = tensors["intrinsics"]
    _require(bool((intr[..., 0, 0] > 0).all() and (intr[..., 1, 1] > 0).all()), "intrinsics: focal lengths must be positive")
    _require(bool((intr[..., 2, :2].abs() <= 1e-6).all() and ((intr[..., 2, 2] - 1).abs() <= 1e-6).all()),
             "intrinsics: expected homogeneous last row [0,0,1]")
    _require(bool((tensors["geometry_confidence"] >= 0).all()), "geometry_confidence: must be nonnegative")
    tokens = _fields(x["patch_tokens"], (), "patch_tokens")
    _require(set(tokens) == set(TOKEN_LAYERS) and all(type(k) is int for k in tokens), "patch_tokens: expected integer layer keys 4,11,17,23")
    n = (h // spec.patch_size) * (w // spec.patch_size)
    for layer in TOKEN_LAYERS:
        name = f"patch_tokens.{layer}"
        tensors[name] = _tensor(tokens[layer], name, (b, v, n, spec.token_dim), frozen=True, device=images.device)
    g = _fields(x["gaussians"], (), "gaussians")
    for name, tensor in _payload_tensors(g, "gaussians"):
        _require(tensor.dtype in FLOAT_DTYPES + (torch.int64, torch.int32, torch.bool), f"{name}: unsupported dtype")
        tensors[name] = _tensor(tensor, name, dtype=tensor.dtype, frozen=True, device=images.device)
    _metadata(x["metadata"], tensors, b, spec)
    if x["metadata"]["evidence_kind"] == "real":
        _require(any(k != "source_indices" for k in g), "gaussians: real caches require A's raw Gaussian payload")
    if x["metadata"]["gaussian_filtered"]:
        _require("source_indices" in g, "gaussians.source_indices: required after filtering")
    if "source_indices" in g:
        _sequence(g["source_indices"], b, "gaussians.source_indices")
        for i, indices in enumerate(g["source_indices"]):
            name = f"gaussians.source_indices[{i}]"
            indices = _tensor(indices, name, dtype=torch.int64, frozen=True, device=images.device)
            _require(indices.ndim == 1, f"{name}: expected one-dimensional indices")
            _require(bool(((indices >= 0) & (indices < v*h*w)).all()), f"{name}: source pixel index out of range")
            _require(indices.unique().numel() == indices.numel(), f"{name}: duplicate source pixel indices")


def validate_instance_targets(targets: InstanceTargets, *, spec: ContractSpec = DEFAULT_SPEC) -> None:
    t = _fields(targets, ("instance_ids", "valid_mask", "confidence", "scene_id", "frame_ids", "label_source"), "targets")
    ids = _tensor(t["instance_ids"], "instance_ids", dtype=torch.int64, frozen=True)
    _require(ids.ndim == 4 and ids.shape[0] > 0, "instance_ids: expected nonempty [B,V,H,W]")
    shape = (ids.shape[0], spec.views, spec.height, spec.width)
    _tensor(ids, "instance_ids", shape, dtype=torch.int64)
    valid = _tensor(t["valid_mask"], "valid_mask", shape, torch.bool, True, ids.device)
    conf = _tensor(t["confidence"], "confidence", shape, frozen=True, device=ids.device)
    _require(bool((ids >= -1).all()), "instance_ids: only -1, 0 or positive IDs are allowed")
    _require(bool(((conf >= 0) & (conf <= 1)).all()), "confidence: expected values in [0,1]")
    _require(not bool((valid & (ids == -1)).any()), "valid_mask: unknown ID -1 cannot be valid")
    _require(bool((conf[~valid] == 0).all()), "confidence: invalid/ignore pixels must have zero weight")
    _require(t["label_source"] in ("gt", "pseudo"), "label_source: expected gt or pseudo")
    _identifiers(t["scene_id"], t["frame_ids"], ids.shape[0], spec.views, "targets")


def scene_instance_ids(targets: InstanceTargets, *, spec: ContractSpec = DEFAULT_SPEC) -> tuple[Tensor, ...]:
    """Scene-level supervised positive IDs, sorted; empty scenes return empty vectors.

    IDs with no positive-weight trusted pixel across ANY input view are omitted.
    This does not infer cross-view physical identity or perform matching/loss.
    """
    validate_instance_targets(targets, spec=spec)
    ids, valid, conf = targets["instance_ids"], targets["valid_mask"], targets["confidence"]
    return tuple(torch.unique(row[mask & (weights > 0) & (row > 0)], sorted=True)
                 for row, mask, weights in zip(ids, valid, conf))


def validate_instance_prediction(prediction: InstancePrediction, *, spec: ContractSpec = DEFAULT_SPEC) -> None:
    """Finite raw logits; Q is any positive integer, shared across all views.

    Values in [0,1] alone cannot prove sigmoid was applied; callers must preserve
    raw logits. Predictions may require gradients. No probabilities are computed.
    """
    p = _fields(prediction, ("mask_logits", "objectness_logits", "query_features"), "prediction")
    masks = _tensor(p["mask_logits"], "mask_logits")
    _require(masks.ndim == 5 and masks.shape[0] > 0 and masks.shape[1] > 0,
             "mask_logits: expected nonempty [B,Q,V,H,W]")
    b, q = masks.shape[:2]
    _tensor(masks, "mask_logits", (b, q, spec.views, spec.height, spec.width))
    _tensor(p["objectness_logits"], "objectness_logits", (b, q), device=masks.device)
    _tensor(p["query_features"], "query_features", (b, q, spec.query_dim), device=masks.device)


def validate_training_batch(batch: FrozenSceneBatch, targets: InstanceTargets,
                            prediction: InstancePrediction | None = None, *,
                            spec: ContractSpec = DEFAULT_SPEC) -> None:
    """Cross-check batch size, devices, scene and frame order before a future loss."""
    validate_frozen_scene_batch(batch, spec=spec)
    validate_instance_targets(targets, spec=spec)
    _require(batch["images"].shape[0] == targets["instance_ids"].shape[0], "targets: batch size mismatch")
    _require(batch["images"].device == targets["instance_ids"].device, "targets: device mismatch")
    for key in ("scene_id", "frame_ids"):
        lhs, rhs = batch["metadata"][key], targets[key]
        if key == "frame_ids":
            lhs, rhs = [tuple(row) for row in lhs], [tuple(row) for row in rhs]
        _require(list(lhs) == list(rhs), f"targets.{key}: differs from feature cache order")
    if prediction is not None:
        validate_instance_prediction(prediction, spec=spec)
        _require(prediction["mask_logits"].shape[0] == batch["images"].shape[0], "prediction: batch size mismatch")
        _require(prediction["mask_logits"].device == batch["images"].device, "prediction: device mismatch")
