"""GT set supervision with explicit exhaustive-set gating; no optimizer."""
import torch
import torch.nn.functional as F

from uni3r.instance.contracts import DEFAULT_SPEC, InstancePrediction, InstanceTargets
from instance_B.matcher import SceneMatcher
from instance_B.loss_utils import LossConfig, graph_zero, prepare_scene, matched_mask_components


class InstanceCriterion:
    def __init__(self, config=None, *, spec=DEFAULT_SPEC):
        self.config = config or LossConfig.from_json()
        self.spec = spec
        self.matcher = SceneMatcher(self.config, spec=spec)

    def __call__(self, prediction: InstancePrediction, targets: InstanceTargets, *, gt_set_exhaustive=None):
        # The matcher runs shared validators first and changes no requires_grad.
        matches = self.matcher(prediction, targets)
        bsize, q = prediction['objectness_logits'].shape
        if gt_set_exhaustive is None:
            gt_set_exhaustive = [False]*bsize
        if (not isinstance(gt_set_exhaustive, (list, tuple)) or len(gt_set_exhaustive) != bsize
                or any(type(e) is not bool for e in gt_set_exhaustive)):
            raise ValueError('gt_set_exhaustive must be a bool list/tuple with one entry per batch item')
        zero = graph_zero(prediction)
        bces, dices, objects, records = [], [], [], []
        for b, match in enumerate(matches):
            scene = prepare_scene(targets, b)
            k = len(match.target_instance_ids)
            exhaustive = gt_set_exhaustive[b]
            negative = exhaustive and scene.weight_sum > 0 and not scene.excluded_zero_support_ids
            reasons = []
            if exhaustive and scene.excluded_zero_support_ids:
                reasons.append('NEGATIVE_SUPERVISION_DISABLED_ZERO_SUPPORT_INSTANCE')
            if scene.weight_sum == 0:
                reasons.append('NO_SUPERVISED_PIXELS')
            elif k == 0 and not negative:
                reasons.append('NO_INSTANCE_LEVEL_SUPERVISION_PARTIAL_GT')
            if not exhaustive:
                reasons.append('GT_SET_NOT_DECLARED_EXHAUSTIVE')
            lb, ld, lo = zero, zero, zero
            if k:
                bc, dc = matched_mask_components(prediction['mask_logits'][b].flatten(1), scene,
                    match.matched_query_indices, match.matched_target_indices, self.config)
                lb, ld = bc.mean(), dc.mean()
                bces.append(lb); dices.append(ld)
            if k or negative:
                logits = prediction['objectness_logits'][b].to(torch.float64)
                # Scale per-query before summing to avoid unnecessary overflow.
                lo = (F.softplus(-logits[match.matched_query_indices])/q).sum()
                if negative:
                    lo = lo + (F.softplus(logits[match.unmatched_query_indices])*(self.config.alpha_no_object/q)).sum()
                objects.append(lo)
            records.append({'batch_index': b, 'scene_id': match.scene_id, 'K': k,
                'S': scene.weight_sum, 'supervised_pixel_count': scene.supervised_pixel_count,
                'gt_set_exhaustive': exhaustive, 'negative_supervision_enabled': bool(negative),
                'excluded_zero_support_ids': scene.excluded_zero_support_ids,
                'mask_active': bool(k), 'objectness_active': bool(k or negative),
                'status': 'SUPERVISED' if k or negative else 'SKIPPED', 'reasons': reasons,
                'loss_bce': lb.detach().item(), 'loss_dice': ld.detach().item(),
                'loss_objectness': lo.detach().item()})
        lb = torch.stack(bces).mean() if bces else zero
        ld = torch.stack(dices).mean() if dices else zero
        lo = torch.stack(objects).mean() if objects else zero
        total = self.config.bce_weight*lb + self.config.dice_weight*ld + self.config.objectness_weight*lo
        if not all(bool(torch.isfinite(v)) for v in (lb, ld, lo, total)):
            raise ValueError('Nonfinite loss reduction')
        return {'loss_total': total, 'loss_bce': lb, 'loss_dice': ld, 'loss_objectness': lo,
                'matches': matches, 'mask_active_scene_count': len(bces),
                'objectness_active_scene_count': len(objects),
                'supervised_scene_count': sum(r['mask_active'] or r['objectness_active'] for r in records),
                'scenes': records, 'scalar_dtype': str(total.dtype)}
