"""Exact scene-global set assignment over all source views; GT-only B-02."""
from dataclasses import dataclass

import scipy
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from uni3r.instance.contracts import (
    DEFAULT_SPEC, ContractError, InstancePrediction, InstanceTargets,
    validate_instance_prediction, validate_instance_targets,
)
from instance_B.loss_utils import LossConfig, prepare_scene, grouped_cost_components

BACKEND = 'scipy.optimize.linear_sum_assignment'
BACKEND_VERSION = scipy.__version__


class CapacityError(ValueError):
    pass


@dataclass
class SceneMatch:
    batch_index: int
    scene_id: str
    target_instance_ids: list[int]
    matched_query_indices: list[int]
    matched_target_indices: list[int]
    matched_instance_ids: list[int]
    unmatched_query_indices: list[int]
    supervised_pixel_count: int
    weight_sum: float
    excluded_zero_support_ids: list[int]
    status: str
    skip_reason: str | None
    cost_finite: bool
    optimal_cost: float
    bce_cost: torch.Tensor
    dice_cost: torch.Tensor
    objectness_cost: torch.Tensor
    total_cost: torch.Tensor
    workspace: dict


@torch.no_grad()
def solve_assignment(cost):
    """Exact rectangular linear assignment. Ties need not pick unique queries."""
    matrix = cost.detach().to(device='cpu', dtype=torch.float64)
    if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all()):
        raise ValueError('Assignment expects finite [Q,K] cost')
    q, k = matrix.shape
    if k > q:
        raise CapacityError(f'Assignment capacity K={k} > Q={q}')
    if k == 0:
        return [], []
    rows, columns = linear_sum_assignment(matrix.numpy())
    return rows.tolist(), columns.tolist()


class SceneMatcher:
    def __init__(self, config=None, *, spec=DEFAULT_SPEC):
        self.config = config or LossConfig.from_json()
        self.spec = spec

    @torch.no_grad()
    def __call__(self, prediction: InstancePrediction, targets: InstanceTargets):
        validate_instance_targets(targets, spec=self.spec)
        validate_instance_prediction(prediction, spec=self.spec)
        if targets['label_source'] != 'gt':
            raise ValueError(f"GT_ONLY: unsupported label_source={targets['label_source']!r}; pseudo distillation is not supported")
        z = prediction['mask_logits']
        if z.shape[0] != targets['instance_ids'].shape[0] or z.device != targets['instance_ids'].device:
            raise ContractError('prediction/targets: batch size or device mismatch')
        matches = []
        for b in range(z.shape[0]):
            scene = prepare_scene(targets, b)
            q, k = z.shape[1], len(scene.target_instance_ids)
            if k > q:
                raise CapacityError(f'CAPACITY_EXCEEDED batch_index={b} scene_id={targets["scene_id"][b]!r} K={k} Q={q}')
            bce, dice, design = grouped_cost_components(z[b].flatten(1), scene, self.config)
            obj = F.softplus(-prediction['objectness_logits'][b].detach().to(device='cpu', dtype=torch.float64))
            cost = self.config.bce_weight*bce + self.config.dice_weight*dice + self.config.objectness_weight*obj[:, None]
            if not bool(torch.isfinite(cost).all() and torch.isfinite(obj).all()):
                raise ValueError(f'Nonfinite matching cost for batch_index={b}')
            qi, ti = solve_assignment(cost)
            reason = 'NO_SUPERVISED_PIXELS' if scene.weight_sum == 0 else ('EMPTY_TARGET_SET' if k == 0 else None)
            matches.append(SceneMatch(b, targets['scene_id'][b], scene.target_instance_ids,
                qi, ti, [scene.target_instance_ids[j] for j in ti],
                [i for i in range(q) if i not in qi], scene.supervised_pixel_count,
                scene.weight_sum, scene.excluded_zero_support_ids,
                'MATCHED' if k else 'EMPTY_MATCH', reason, True,
                cost[qi, ti].sum().item(), bce, dice, obj, cost, design))
        return matches
