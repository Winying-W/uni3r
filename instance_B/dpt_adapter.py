"""B-owned, patch-only adapter around the checkout's actual Uni3R DPT class."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys

import torch
from torch import Tensor, nn

from uni3r.instance.contracts import TOKEN_LAYERS

ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = ROOT / "submodules/vggt"

# Only make this checkout's vendored dependency importable; no model constructor,
# environment changes, downloads or global class/parameter monkey patches.
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))
from uni3r.dpt_head import DPTHead as OriginalDPTHead  # noqa: E402
import vggt.heads.dpt_head as _vendored_dpt  # noqa: E402

if Path(_vendored_dpt.__file__).resolve() != (VGGT_ROOT / "vggt/heads/dpt_head.py").resolve():
    raise ImportError("DPT dependency was imported from a different checkout")


def make_reference_dpt() -> OriginalDPTHead:
    """Construct only the original DPT, with the original 24-list indices."""
    return OriginalDPTHead(dim_in=2048, feature_only=True, input_identity=True, patch_size=16)


def strict_load_dpt(module: nn.Module, state: Mapping[str, Tensor]) -> dict:
    """Precheck every key/shape/value, then copy without parameter/storage sharing."""
    expected = module.state_dict()
    missing, extra = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError(f"DPT keys mismatch: missing={missing}, unexpected={extra}")
    for key, value in state.items():
        if not isinstance(value, Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"DPT shape mismatch: {key}")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"DPT invalid dtype/nonfinite value: {key}")
    result = module.load_state_dict(state, strict=True)
    module.requires_grad_(True)
    for key, value in module.state_dict().items():
        if not torch.equal(value.cpu(), state[key].to(dtype=value.dtype, device="cpu")):
            raise RuntimeError(f"DPT copy verification failed: {key}")
        if value.device == state[key].device and value.data_ptr() == state[key].data_ptr():
            raise RuntimeError(f"DPT storage unexpectedly shared: {key}")
    return {"keys": len(expected), "parameters": sum(p.numel() for p in module.parameters()),
            "missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
            "all_trainable": all(p.requires_grad for p in module.parameters()), "strict": True}


class PatchOnlyDPT(OriginalDPTHead):
    """Same operations and weights; adapt only this instance to four cached layers.

    No fake 24-element list and no token deletion. The four-element list passed to
    the inherited implementation is explicitly indexed by this instance's [0..3].
    RGB remains [-1,1], matching Uni3R's actual input_identity call path.
    """
    def __init__(self):
        super().__init__(dim_in=2048, feature_only=True, input_identity=True,
                         patch_size=16, intermediate_layer_idx=[0, 1, 2, 3])

    def forward(self, patch_tokens: Mapping[int, Tensor], images: Tensor) -> Tensor:
        if set(patch_tokens) != set(TOKEN_LAYERS):
            raise ValueError("PatchOnlyDPT requires exactly layers [4,11,17,23]")
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("DPT images must be [B,V,3,H,W]")
        b, v, _, h, w = images.shape
        if h % 16 or w % 16:
            raise ValueError("DPT image size must be divisible by 16")
        expected = (b, v, (h // 16) * (w // 16), 2048)
        ordered = []
        for layer in TOKEN_LAYERS:
            if tuple(patch_tokens[layer].shape) != expected:
                raise ValueError(f"DPT layer {layer}: expected patch-only shape {expected}")
            # Detach inputs, never the DPT computation or trainable parameters.
            ordered.append(patch_tokens[layer].detach().contiguous())
        return super().forward(ordered, images.detach().contiguous(), patch_start_idx=0,
                               frames_chunk_size=None)
