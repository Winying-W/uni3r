"""B-private exact reductions. Shared contracts remain the only input schema."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    bce_weight: float
    dice_weight: float
    objectness_weight: float
    eps: float
    alpha_no_object: float
    chunk_size: int
    workspace_budget_bytes: int

    def __post_init__(self):
        for key in ('bce_weight', 'dice_weight', 'objectness_weight'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key}: expected positive finite weight')
        if self.eps != 1e-6 or self.alpha_no_object != 0.1:
            raise ValueError('B-02 fixes eps=1e-6 and alpha_no_object=0.1')
        for key in ('chunk_size', 'workspace_budget_bytes'):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f'{key}: expected positive integer')
        if self.workspace_budget_bytes > 64 * 1024**2:
            raise ValueError('Cost workspace design budget cannot exceed 64 MiB')

    @classmethod
    def from_json(cls, path=None):
        path = Path(path) if path else Path(__file__).parent / 'configs/loss_v0.json'
        return cls(**json.loads(path.read_text()))


@dataclass
class PreparedScene:
    """Internal reduction state; never an alternative external target format."""
    target_instance_ids: list[int]
    excluded_zero_support_ids: list[int]
    pixel_indices: torch.Tensor
    groups: torch.Tensor
    u: torch.Tensor
    weight_sum: float
    supervised_pixel_count: int
    source_pixel_count: int


@torch.no_grad()
def prepare_scene(targets, batch_index):
    ids = targets['instance_ids'][batch_index].detach().reshape(-1).cpu()
    weights = targets['confidence'][batch_index].detach().reshape(-1).to(device='cpu', dtype=torch.float64)
    weights = weights * targets['valid_mask'][batch_index].detach().reshape(-1).cpu()
    pixels = torch.nonzero(weights > 0, as_tuple=False).flatten()
    active_ids = torch.unique(ids[pixels][ids[pixels] > 0], sorted=True)
    all_ids = torch.unique(ids[ids > 0], sorted=True).tolist()
    positive_ids = active_ids.tolist()
    excluded = sorted(set(all_ids) - set(positive_ids))
    total = weights.sum().item()
    # No divide on S=0. Normalize across all views, in float64.
    u = weights[pixels] / total if total > 0 else weights[pixels]
    labels = ids[pixels]
    groups = torch.zeros_like(labels)
    foreground = labels > 0
    groups[foreground] = torch.searchsorted(active_ids, labels[foreground]) + 1
    return PreparedScene(positive_ids, excluded, pixels, groups, u, total, pixels.numel(), ids.numel())


def workspace_design(q, k, n, chunk_size):
    """Conservative live tensor envelope, not allocator/RSS/operator measurement.

    Twelve QxC float64 buffers exceed this implementation's live chunk buffers.
    Covers source/target bookkeeping and grouped/prefix/suffix/cost matrices too.
    Original predictions, student state and allocator overhead are excluded.
    """
    c = min(n, chunk_size)
    estimate = 12*q*c*8 + 16*q*(k+1)*8 + 64*n
    return {'largest_chunk_shape': [q, c], 'chunk_dtype': 'torch.float64',
            'largest_chunk_bytes': q*c*8, 'conservative_workspace_bytes': estimate,
            'group_matrix_shape': [q, k+1], 'no_Q_K_N_tensor': True}


@torch.no_grad()
def grouped_cost_components(flat_logits, scene, config):
    """Exact QxK mask matrices with only Qxchunk and Qx(K+1) work tensors."""
    q = flat_logits.shape[0]
    k = len(scene.target_instance_ids)
    design = workspace_design(q, k, scene.source_pixel_count, config.chunk_size)
    if design['conservative_workspace_bytes'] > config.workspace_budget_bytes:
        raise ValueError(f'Cost workspace design exceeds budget: {design}')
    neg = torch.zeros(q, k+1, dtype=torch.float64)
    pos = torch.zeros_like(neg)
    mass = torch.zeros_like(neg)
    target_mass = torch.zeros(k+1, dtype=torch.float64)
    for start in range(0, scene.supervised_pixel_count, config.chunk_size):
        sl = slice(start, start+config.chunk_size)
        indices = scene.pixel_indices[sl].to(flat_logits.device)
        z = flat_logits.index_select(1, indices).to(device='cpu', dtype=torch.float64)
        u, g = scene.u[sl], scene.groups[sl]
        index = g.unsqueeze(0).expand(q, -1)
        neg.scatter_add_(1, index, F.softplus(z) * u)
        pos.scatter_add_(1, index, F.softplus(-z) * u)
        mass.scatter_add_(1, index, z.sigmoid() * u)
        target_mass.scatter_add_(0, g, u)
    # Exclusive prefix/suffix sums avoid subtracting near-equal large totals.
    zeros = torch.zeros(q, 1, dtype=torch.float64)
    before = torch.cat((zeros, neg[:, :-1].cumsum(1)), dim=1)
    after = torch.cat((neg[:, 1:].flip(1).cumsum(1).flip(1), zeros), dim=1)
    bce = pos[:, 1:] + (before+after)[:, 1:]
    dice = 1 - (2*mass[:, 1:]+config.eps) / (mass.sum(1, keepdim=True)+target_mass[1:]+config.eps)
    if not bool(torch.isfinite(bce).all() and torch.isfinite(dice).all()):
        raise ValueError('Nonfinite mask cost reduction')
    return bce, dice, design


def graph_zero(prediction):
    # Slice before reduction: summing huge finite logits then multiplying by zero
    # could overflow. Both raw output branches stay connected to the graph.
    return sum(v.reshape(-1)[:1].to(torch.float64).mul(0).sum()
               for v in (prediction['mask_logits'], prediction['objectness_logits']))


def matched_mask_components(flat_logits, scene, query_indices, target_indices, config):
    """Differentiable matched pairs only; no detached prediction or QxKxN graph."""
    device = flat_logits.device
    count = len(query_indices)
    qidx = torch.tensor(query_indices, device=device, dtype=torch.long)
    groups = torch.tensor(target_indices, device=device, dtype=torch.long) + 1
    bce = torch.zeros(count, device=device, dtype=torch.float64)
    pred_mass, intersection, target_mass = bce.clone(), bce.clone(), bce.clone()
    for start in range(0, scene.supervised_pixel_count, config.chunk_size):
        sl = slice(start, start+config.chunk_size)
        pixels = scene.pixel_indices[sl].to(device)
        # Advanced indexing materializes only [matched_pairs, chunk].
        z = flat_logits[qidx[:, None], pixels[None, :]].to(torch.float64)
        u = scene.u[sl].to(device)
        t = (groups[:, None] == scene.groups[sl].to(device)[None, :]).to(torch.float64)
        bce = bce + ((t*F.softplus(-z)+(1-t)*F.softplus(z))*u).sum(1)
        p = z.sigmoid()
        pred_mass = pred_mass + (p*u).sum(1)
        intersection = intersection + (p*t*u).sum(1)
        target_mass = target_mass + (t*u).sum(1)
    dice = 1 - (2*intersection+config.eps)/(pred_mass+target_mass+config.eps)
    return bce, dice
