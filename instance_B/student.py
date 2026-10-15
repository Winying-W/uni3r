"""Independent Uni3R instance student. No targets, teacher, matching or loss."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from uni3r.instance.contracts import (
    CONTRACT_VERSION, TOKEN_LAYERS, ContractSpec, FrozenSceneBatch, InstancePrediction,
    validate_frozen_scene_batch, validate_instance_prediction,
)
from instance_B.dpt_adapter import ROOT, PatchOnlyDPT, strict_load_dpt

BASE_SHA256 = "660aed13d6a8bccd078be35210f803d0f6d3b312f63dd468bbccc1c7f3aacf24"


@dataclass(frozen=True)
class StudentConfig:
    height: int = 256
    width: int = 256
    views: int = 2
    token_layers: tuple[int, ...] = TOKEN_LAYERS
    token_dim: int = 2048
    patch_size: int = 16
    num_queries: int = 100
    hidden_dim: int = 256
    decoder_layers: int = 6
    attention_heads: int = 8
    feedforward_dim: int = 1024
    mask_feature_dim: int = 64
    memory_strides: tuple[int, ...] = (8, 16)
    max_memory_tokens: int = 8192
    xyz_epsilon: float = 1e-6
    xyz_normalization: str = "per_scene_mean_rms_float64"
    geometry_confidence_transform: str = "log1p_nonnegative_score"
    rgb_normalization: str = "minus_one_to_one"
    dropout: float = 0.0
    init_mode: str = "pretrained_2views"
    profile: str = "default_v0"

    def __post_init__(self):
        integers = ("height", "width", "views", "token_dim", "patch_size", "num_queries",
                    "hidden_dim", "decoder_layers", "attention_heads", "feedforward_dim",
                    "mask_feature_dim", "max_memory_tokens")
        for key in integers:
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"StudentConfig.{key} must be a positive integer")
        if (self.views, self.token_dim, self.patch_size, self.hidden_dim, self.mask_feature_dim) != (2, 2048, 16, 256, 64):
            raise ValueError("v0 retains V=2, token_dim=2048, patch=16, hidden=256, mask_dim=64")
        if tuple(self.token_layers) != TOKEN_LAYERS:
            raise ValueError("StudentConfig requires token layers [4,11,17,23]")
        if self.height % 16 or self.width % 16 or self.hidden_dim % self.attention_heads:
            raise ValueError("Invalid image size or attention head divisibility")
        if self.profile not in ("default_v0", "synthetic_small"):
            raise ValueError("Unknown student profile")
        if self.profile == "default_v0" and (self.height, self.width, self.decoder_layers, self.attention_heads) != (256, 256, 6, 8):
            raise ValueError("Default profile dimensions/depth are fixed; select synthetic_small explicitly")
        if not self.memory_strides or any(type(s) is not int or s < 1 or self.height % s or self.width % s for s in self.memory_strides):
            raise ValueError("Memory strides must divide image dimensions")
        if len(set(self.memory_strides)) != len(self.memory_strides):
            raise ValueError("Duplicate memory scales")
        if self.memory_token_count > self.max_memory_tokens or self.max_memory_tokens > 8192:
            raise ValueError("Cross-view multiscale memory exceeds the 8192 token budget")
        if not math.isfinite(self.xyz_epsilon) or self.xyz_epsilon <= 0:
            raise ValueError("xyz_epsilon must be finite and positive")
        if self.xyz_normalization != "per_scene_mean_rms_float64" or self.geometry_confidence_transform != "log1p_nonnegative_score":
            raise ValueError("Unsupported geometry transform")
        if self.rgb_normalization != "minus_one_to_one":
            raise ValueError("Original Uni3R DPT input_identity expects [-1,1] RGB")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")
        if self.init_mode not in ("pretrained_2views", "synthetic_random_init"):
            raise ValueError("Unknown initialization mode")

    @property
    def memory_token_count(self) -> int:
        return self.views * sum((self.height // s) * (self.width // s) for s in self.memory_strides)

    @property
    def contract_spec(self) -> ContractSpec:
        return ContractSpec(height=self.height, width=self.width, views=self.views,
                            token_dim=self.token_dim, patch_size=self.patch_size, query_dim=self.hidden_dim)

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text())
        for key in ("token_layers", "memory_strides"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DPTInitialization:
    state: Mapping[str, Tensor]
    receipt: dict


def load_dpt_initialization(weights_path: str | Path, receipt_path: str | Path) -> DPTInitialization:
    """Read a small tensor-only artifact and verify its extraction receipt/hash."""
    weights, receipt_file = Path(weights_path).resolve(), Path(receipt_path).resolve()
    if not all(p.is_relative_to(ROOT / "instance_B") for p in (weights, receipt_file)):
        raise ValueError("DPT initialization artifacts must be under instance_B")
    receipt = json.loads(receipt_file.read_text())
    if receipt.get("status") != "DPT_INIT_VERIFIED" or receipt.get("source_checkpoint_sha256") != BASE_SHA256:
        raise ValueError("Unverified DPT initialization source")
    if receipt.get("purpose") != "INITIALIZATION_ONLY_NOT_TRAINED_STUDENT":
        raise ValueError("Incorrect DPT artifact purpose")
    if weights.stat().st_size > 512 * 1024**2:
        raise ValueError("Unexpectedly large DPT artifact")
    data = weights.read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt.get("artifact_sha256"):
        raise ValueError("DPT initialization artifact SHA256 mismatch")
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, Tensor) for k,v in state.items()):
        raise ValueError("Expected a tensor-only DPT state_dict")
    return DPTInitialization(state=state, receipt=receipt)


def normalize_predicted_xyz(means: Tensor, epsilon: float) -> tuple[Tensor, Tensor, Tensor]:
    """Per scene, over all V,H,W: center=arithmetic mean, scale=RMS radius.

    Float64 reductions avoid float32 overflow; clamp scale below epsilon only.
    No confidence weighting, no GT coordinates, no cross-batch normalization.
    """
    xyz = means.detach().to(torch.float64)
    center = xyz.mean(dim=(1, 2, 3), keepdim=True)
    centered = xyz - center
    scale = centered.square().sum(-1, keepdim=True).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(epsilon)
    return (centered / scale).to(dtype=means.dtype), center, scale


class GeometryPositionEncoder(nn.Module):
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.xy = nn.Linear(2, d)
        self.xyz = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.confidence = nn.Linear(1, d)
        self.view = nn.Embedding(config.views, d)
        self.level = nn.Embedding(len(config.memory_strides), d)

    def forward(self, normalized_xyz: Tensor, confidence: Tensor, size: tuple[int, int], level: int) -> Tensor:
        b, v, h, w, _ = normalized_xyz.shape
        mh, mw = size
        xyz = normalized_xyz.permute(0, 1, 4, 2, 3).reshape(b*v, 3, h, w)
        xyz = F.adaptive_avg_pool2d(xyz, size).reshape(b, v, 3, mh, mw).permute(0, 1, 3, 4, 2)
        # Confidence is an unbounded nonnegative model score. Apply log1p before
        # area pooling; never clamp to [0,1] or interpret it as a probability.
        conf = torch.log1p(confidence.detach()).reshape(b*v, 1, h, w)
        conf = F.adaptive_avg_pool2d(conf, size).reshape(b, v, mh, mw, 1)
        yy = (torch.arange(mh, device=xyz.device, dtype=xyz.dtype) + 0.5) * (2 / mh) - 1
        xx = (torch.arange(mw, device=xyz.device, dtype=xyz.dtype) + 0.5) * (2 / mw) - 1
        y, x = torch.meshgrid(yy, xx, indexing="ij")
        xy = torch.stack((x, y), dim=-1)
        view = self.view(torch.arange(v, device=xyz.device))[None, :, None, None, :]
        position = self.xy(xy)[None, None] + self.xyz(xyz) + self.confidence(conf) + view + self.level.weight[level]
        return position.reshape(b, v*mh*mw, self.config.hidden_dim)


class MultiScaleAdapter(nn.Module):
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        self.projections = nn.ModuleList(nn.Conv2d(256, config.hidden_dim, 1) for _ in config.memory_strides)
        self.norms = nn.ModuleList(nn.LayerNorm(config.hidden_dim) for _ in config.memory_strides)

    def forward(self, dense: Tensor) -> tuple[list[Tensor], list[tuple[int, int]]]:
        b, v, c, h, w = dense.shape
        memories, shapes = [], []
        for stride, proj, norm in zip(self.config.memory_strides, self.projections, self.norms):
            size = (h // stride, w // stride)
            value = proj(F.adaptive_avg_pool2d(dense.reshape(b*v, c, h, w), size))
            value = value.reshape(b, v, self.config.hidden_dim, *size).permute(0, 1, 3, 4, 2)
            memories.append(norm(value.reshape(b, v*size[0]*size[1], self.config.hidden_dim)))
            shapes.append(size)
        return memories, shapes


class QueryDecoderLayer(nn.Module):
    """Only QxQ self-attention and QxM cross-attention; no MxM attention."""
    def __init__(self, config: StudentConfig):
        super().__init__()
        d = config.hidden_dim
        self.self_attention = nn.MultiheadAttention(d, config.attention_heads, dropout=config.dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d, config.attention_heads, dropout=config.dropout, batch_first=True)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, config.feedforward_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.feedforward_dim, d))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: Tensor, memory: Tensor, position: Tensor) -> Tensor:
        q = self.norm1(query)
        query = query + self.dropout(self.self_attention(q, q, q, need_weights=False)[0])
        query = query + self.dropout(self.cross_attention(self.norm2(query), memory + position, memory, need_weights=False)[0])
        return query + self.dropout(self.ffn(self.norm3(query)))


def mask_logits_from_features(query_masks: Tensor, pixels: Tensor) -> Tensor:
    """[B,Q,64] dot [B,V,64,H,W], divided by sqrt(64), without activation."""
    if query_masks.ndim != 3 or pixels.ndim != 5 or query_masks.shape[0] != pixels.shape[0] or query_masks.shape[-1] != 64 or pixels.shape[2] != 64:
        raise ValueError("Mask features must use a shared 64D embedding")
    return torch.einsum("bqc,bvchw->bqvhw", query_masks, pixels) / math.sqrt(64)


def flatten_membership_logits(mask_logits: Tensor, source_indices: list[Tensor] | None = None):
    """Pure view-y-x indexing, with optional per-scene filtered source indices.

    No instance thresholding/clustering/rendering. Filtering returns a list of
    [Q,Mi] tensors so each scene may have a different number of retained Gaussians.
    """
    if mask_logits.ndim != 5 or not mask_logits.is_floating_point() or not bool(torch.isfinite(mask_logits).all()):
        raise ValueError("Expected finite floating [B,Q,V,H,W] mask_logits")
    flattened = mask_logits.flatten(2)
    if source_indices is None:
        return flattened
    if len(source_indices) != mask_logits.shape[0]:
        raise ValueError("source_indices must have one vector per scene")
    result = []
    for scene, indices in zip(flattened, source_indices):
        if not isinstance(indices, Tensor) or indices.dtype != torch.int64 or indices.ndim != 1 or indices.device != scene.device:
            raise ValueError("source_indices must be same-device int64 vectors")
        if bool(((indices < 0) | (indices >= scene.shape[-1])).any()) or indices.unique().numel() != indices.numel():
            raise ValueError("source_indices has out-of-range or duplicate entries")
        result.append(scene.index_select(-1, indices))
    return result


class InstanceStudent(nn.Module):
    def __init__(self, config: StudentConfig | None = None, *, dpt_init: DPTInitialization | None = None):
        super().__init__()
        self.config = config or StudentConfig()
        if self.config.init_mode == "pretrained_2views":
            if dpt_init is None or dpt_init.receipt.get("status") != "DPT_INIT_VERIFIED" or dpt_init.receipt.get("source_checkpoint_sha256") != BASE_SHA256:
                raise ValueError("DPT_INIT_BLOCKED: verified 2views initialization is required; no random fallback")
        elif dpt_init is not None:
            raise ValueError("synthetic_random_init cannot claim pretrained initialization")
        self.instance_dpt = PatchOnlyDPT()
        self.initialization = (strict_load_dpt(self.instance_dpt, dpt_init.state) if dpt_init else {"mode": "synthetic_random_init"})
        self.adapter = MultiScaleAdapter(self.config)
        self.position_encoder = GeometryPositionEncoder(self.config)
        self.object_queries = nn.Parameter(torch.randn(self.config.num_queries, self.config.hidden_dim) * 0.02)
        self.decoder = nn.ModuleList(QueryDecoderLayer(self.config) for _ in range(self.config.decoder_layers))
        self.query_norm = nn.LayerNorm(self.config.hidden_dim)
        self.mask_features = nn.Conv2d(256, self.config.mask_feature_dim, 1)
        self.query_mask_embedding = nn.Linear(self.config.hidden_dim, self.config.mask_feature_dim)
        self.objectness_head = nn.Linear(self.config.hidden_dim, 1)

    def forward(self, batch: FrozenSceneBatch) -> InstancePrediction:
        validate_frozen_scene_batch(batch, spec=self.config.contract_spec)
        if self.config.init_mode == "synthetic_random_init" and batch["metadata"]["evidence_kind"] != "synthetic":
            raise ValueError("synthetic_random_init cannot consume real caches")
        if self.config.profile == "synthetic_small" and batch["metadata"]["evidence_kind"] != "synthetic":
            raise ValueError("synthetic_small cannot consume real caches")
        if batch["metadata"]["rgb_normalization"] != self.config.rgb_normalization:
            raise ValueError("RGB metadata differs from DPT's [-1,1] input_identity path")
        parameter = self.object_queries
        needed = [batch["images"], batch["means"], batch["geometry_confidence"], *batch["patch_tokens"].values()]
        if any(t.dtype != torch.float32 or t.device != parameter.device for t in needed) or parameter.dtype != torch.float32:
            raise ValueError("B-01 student supports same-device float32 inputs/parameters; AMP not yet validated")
        dense = self.instance_dpt(batch["patch_tokens"], batch["images"])
        b, v, c, h, w = dense.shape
        memories, shapes = self.adapter(dense)
        xyz, _, _ = normalize_predicted_xyz(batch["means"], self.config.xyz_epsilon)
        positions = [self.position_encoder(xyz, batch["geometry_confidence"], size, i) for i, size in enumerate(shapes)]
        memory, position = torch.cat(memories, dim=1), torch.cat(positions, dim=1)
        if memory.shape[1] != self.config.memory_token_count or memory.shape[1] > self.config.max_memory_tokens:
            raise RuntimeError("Unexpected memory token count")
        query = self.object_queries.unsqueeze(0).expand(b, -1, -1)
        for layer in self.decoder:
            query = layer(query, memory, position)
        query = self.query_norm(query)
        pixels = self.mask_features(dense.reshape(b*v, c, h, w)).reshape(b, v, 64, h, w)
        prediction = {"mask_logits": mask_logits_from_features(self.query_mask_embedding(query), pixels),
                      "objectness_logits": self.objectness_head(query).squeeze(-1), "query_features": query}
        validate_instance_prediction(prediction, spec=self.config.contract_spec)
        return prediction
