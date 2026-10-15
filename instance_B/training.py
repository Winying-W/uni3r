"""Minimal CPU synthetic training engine; separate active-scene denominators."""
from dataclasses import dataclass, asdict
import fcntl
import json
import math
import os
from pathlib import Path
import time

import torch

from instance_B.b03_utils import B03_ROOT, canonical, project_path
from instance_B.criterion import InstanceCriterion
from instance_B.loss_utils import LossConfig, prepare_scene, graph_zero
from instance_B.matcher import CapacityError
from instance_B.student import InstanceStudent
from uni3r.instance.contracts import validate_training_batch

PURPOSE = 'SYNTHETIC_ENGINE_DIAGNOSTIC_NOT_FOR_REAL_TRAINING'


@dataclass(frozen=True)
class TrainConfig:
    purpose: str
    training_track: str
    optimizer: str
    dpt_lr: float
    new_lr: float
    weight_decay: float
    betas: list
    eps: float
    amsgrad: bool
    foreach: bool
    fused: bool
    max_grad_norm: float
    max_window_scenes: int
    micro_batch_scenes: int
    lr_mode: str
    normalization: str
    checkpoint_window_boundary_only: bool
    physical_step_limit: int
    seed: int

    def __post_init__(self):
        expected = dict(purpose=PURPOSE, training_track='synthetic_engine_diagnostic', optimizer='AdamW',
            dpt_lr=1e-5, new_lr=1e-4, weight_decay=.05, betas=[.9, .999], eps=1e-8,
            amsgrad=False, foreach=False, fused=False, max_grad_norm=1., max_window_scenes=4,
            micro_batch_scenes=1, lr_mode='constant', normalization='separate_mask_and_objectness_active_scene_counts',
            checkpoint_window_boundary_only=True, physical_step_limit=8, seed=7303)
        if canonical(asdict(self)) != expected:
            raise ValueError('B-03 diagnostic training configuration is frozen')

    @classmethod
    def from_json(cls, path=None):
        p = Path(path) if path else Path(__file__).parent/'configs/train_smoke_v0.json'
        return cls(**json.loads(project_path(p).read_text()))


class BudgetLedger:
    """One persistent stage ledger, external to every checkpoint and run state.

    A reservation is charged even if a process dies before completion. Ambiguous
    unfinished reservations block further calls; checkpoint restore cannot reset it.
    Linux flock serializes processes; a second stage ledger is rejected.
    """
    def __init__(self, path):
        self.path = project_path(path, write=True)
        if self.path.name != 'optimizer_budget.jsonl' or self.path.parent.parent != B03_ROOT.resolve():
            raise ValueError('Expected a run-root B-03 optimizer_budget.jsonl')
        self._check_canonical()

    def _check_canonical(self):
        paths = [p.resolve() for p in B03_ROOT.glob('*/optimizer_budget.jsonl')]
        if paths != [self.path] and set(paths) != {self.path}:
            raise ValueError('Ambiguous or missing stage budget ledger; never reset budget')

    def _records(self, stream):
        stream.seek(0)
        records = [json.loads(line) for line in stream if line.strip()]
        if not records or records[0] != {'event': 'HEADER', 'work_package': 'B-03', 'limit': 8, 'checkpoint_independent': True}:
            raise ValueError('Invalid immutable B-03 budget header')
        return records

    def records(self):
        self._check_canonical()
        with self.path.open('r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            return self._records(f)

    def reserve(self, context):
        self._check_canonical()
        with self.path.open('r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            records = self._records(f)
            reserved = [r['call_id'] for r in records if r['event'] == 'RESERVED']
            finished = {r['call_id'] for r in records if r['event'] in ('SUCCEEDED', 'FAILED')}
            if set(reserved)-finished: raise RuntimeError('Unresolved physical call; manual review required')
            if len(reserved) >= 8: raise RuntimeError('B-03 physical optimizer budget exhausted (8)')
            call = len(reserved)+1
            self._append(f, {'event': 'RESERVED', 'call_id': call, 'context': context, 'time_ns': time.time_ns()})
            return call

    @staticmethod
    def _append(stream, record):
        stream.seek(0, os.SEEK_END); stream.write(json.dumps(record, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())

    def event(self, event, call_id, **details):
        with self.path.open('r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            records = self._records(f)
            if not any(r.get('call_id') == call_id and r['event'] == 'RESERVED' for r in records):
                raise ValueError('Call was not reserved')
            self._append(f, {'event': event, 'call_id': call_id, **details, 'time_ns': time.time_ns()})

    @property
    def physical_calls(self):
        return sum(r['event'] == 'CALLING' for r in self.records())


class BudgetedAdamW(torch.optim.AdamW):
    def __init__(self, groups, config, ledger):
        self.ledger = ledger
        self.call_context = None
        super().__init__(groups, betas=tuple(config.betas), eps=config.eps, weight_decay=config.weight_decay,
                         amsgrad=config.amsgrad, foreach=config.foreach, fused=config.fused)

    def step(self, closure=None):
        if closure is not None or self.call_context is None:
            raise RuntimeError('Only an admitted synthetic engine window may call step')
        context = self.call_context; self.call_context = None
        call = self.ledger.reserve(context)
        self.ledger.event('CALLING', call)
        try:
            result = super().step()
        except BaseException as exc:
            self.ledger.event('FAILED', call, error=repr(exc)); raise
        self.ledger.event('SUCCEEDED', call)
        return result


def parameter_groups(student, config):
    if not isinstance(student, InstanceStudent): raise TypeError('Requires the existing InstanceStudent')
    pairs = sorted((name, p) for name, p in student.named_parameters() if p.requires_grad)
    groups = []
    for dpt in (True, False):
        selected = [(n, p) for n, p in pairs if n.startswith('instance_dpt.') == dpt]
        groups.append({'params': [p for _, p in selected], 'param_names': [n for n, _ in selected],
                       'group_name': 'instance_dpt' if dpt else 'new_modules',
                       'lr': config.dpt_lr if dpt else config.new_lr})
    validate_parameter_groups(student, groups, config)
    return groups


def validate_parameter_groups(student, groups, config):
    expected = dict((n, p) for n, p in student.named_parameters() if p.requires_grad)
    if len(groups) != 2: raise ValueError('Expected exactly DPT and new-module groups')
    names, objects, storages = [], [], []
    for i, g in enumerate(groups):
        wanted = sorted(n for n in expected if n.startswith('instance_dpt.') == (i == 0))
        if g['param_names'] != wanted or len(g['params']) != len(wanted): raise ValueError('Parameter name/order/coverage mismatch')
        if g['lr'] != (config.dpt_lr if i == 0 else config.new_lr): raise ValueError('Wrong group learning rate')
        for name, p in zip(g['param_names'], g['params']):
            if p is not expected[name] or p.device.type != 'cpu' or p.dtype != torch.float32:
                raise ValueError('Parameter object/device/dtype mismatch')
            names.append(name); objects.append(id(p)); storages.append(p.untyped_storage().data_ptr())
    if set(names) != set(expected) or len(names) != len(expected) or len(set(objects)) != len(objects) or len(set(storages)) != len(storages):
        raise ValueError('Duplicate or missing trainable parameter/storage')


def optimizer_group_manifest(optimizer):
    return [canonical({**{k: v for k, v in group.items() if k != 'params'},
        'tensor_count': len(group['params']), 'element_count': sum(p.numel() for p in group['params'])}) for group in optimizer.param_groups]


def gradient_summary(student):
    groups = {}
    for name, p in student.named_parameters():
        if not p.requires_grad: continue
        key = name.split('.')[0]
        rec = groups.setdefault(key, {'tensors': 0, 'with_grad': 0, 'finite': 0, 'nonzero_tensors': 0, 'squared_norm': 0.})
        rec['tensors'] += 1
        if p.grad is not None:
            rec['with_grad'] += 1
            finite = bool(torch.isfinite(p.grad).all()); rec['finite'] += int(finite)
            rec['nonzero_tensors'] += int(bool(torch.count_nonzero(p.grad)))
            if not finite: raise FloatingPointError(f'Nonfinite gradient: {name}')
            rec['squared_norm'] += float(torch.linalg.vector_norm(p.grad.double()))**2
    for rec in groups.values(): rec['norm'] = math.sqrt(rec.pop('squared_norm'))
    return groups


def joint_grad_norm(parameters):
    return math.sqrt(sum(float(torch.linalg.vector_norm(p.grad.double()))**2 for p in parameters if p.grad is not None))


class TrainingEngine:
    def __init__(self, student, ledger, *, loss_config=None, train_config=None, event_log=None):
        self.model = student
        self.config = train_config or TrainConfig.from_json()
        self.loss_config = loss_config or LossConfig.from_json()
        c = student.config
        if (c.profile, c.height, c.width, c.num_queries, c.decoder_layers, c.feedforward_dim, c.dropout) != ('synthetic_small', 32, 32, 7, 2, 512, 0.):
            raise ValueError('B-03 optimizer engine only admits the fixed synthetic_small student')
        if (self.loss_config.bce_weight, self.loss_config.dice_weight, self.loss_config.objectness_weight) != (5., 5., 2.):
            raise ValueError('B-02 loss weights are frozen')
        self.criterion = InstanceCriterion(self.loss_config, spec=c.contract_spec)
        self.optimizer = BudgetedAdamW(parameter_groups(student, self.config), self.config, ledger)
        self.ledger = ledger
        self.event_log = project_path(event_log, write=True) if event_log else None
        self.state = {'global_update_step': 0, 'seen_scenes': 0, 'skipped_scenes': 0,
                      'data_cursor': 0, 'window_cursor': 0, 'failed_windows': 0}
        self.at_boundary = True

    def record(self, event):
        if self.event_log:
            with self.event_log.open('a') as f:
                f.write(json.dumps(event, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())

    @torch.no_grad()
    def preflight(self, window):
        if not isinstance(window, (list, tuple)) or not 1 <= len(window) <= 4:
            raise ValueError('Window must contain 1..4 fixed micro-scenes')
        records = []
        for index, item in enumerate(window):
            if not isinstance(item, (list, tuple)) or len(item) not in (2, 3): raise ValueError('Expected (FrozenSceneBatch, InstanceTargets[, exhaustive])')
            batch, targets = item[:2]; flag = item[2] if len(item) == 3 else False
            if type(flag) is not bool: raise ValueError('exhaustive must be bool')
            validate_training_batch(batch, targets, spec=self.model.config.contract_spec)
            if batch['images'].shape[0] != 1: raise ValueError('Micro-batch must contain one scene')
            if batch['images'].device.type != 'cpu' or batch['metadata']['evidence_kind'] != 'synthetic':
                raise ValueError('REAL_TRAINING_NOT_AUTHORIZED: synthetic CPU inputs only')
            if targets['label_source'] != 'gt': raise ValueError('GT_ONLY: pseudo not supported')
            scene = prepare_scene(targets, 0); k = len(scene.target_instance_ids)
            if k > self.model.config.num_queries:
                raise CapacityError(f'CAPACITY_EXCEEDED micro_index={index} scene_id={targets["scene_id"][0]} K={k} Q={self.model.config.num_queries}')
            negative = flag and scene.weight_sum > 0 and not scene.excluded_zero_support_ids
            records.append({'micro_index': index, 'scene_id': targets['scene_id'][0], 'frame_ids': list(targets['frame_ids'][0]),
                'K': k, 'S': scene.weight_sum, 'mask_active': int(k > 0), 'objectness_active': int(k > 0 or negative),
                'gt_set_exhaustive': flag, 'negative_supervision_enabled': bool(negative),
                'excluded_zero_support_ids': scene.excluded_zero_support_ids})
        return records

    def discard_gradients(self):
        self.optimizer.zero_grad(set_to_none=True); self.at_boundary = True

    def backward_window(self, window):
        if not self.at_boundary: raise RuntimeError('Pending window; middle-window resume/overlap unsupported')
        self.optimizer.zero_grad(set_to_none=True)
        self.at_boundary = False
        try:
            records = self.preflight(window)
            m = sum(r['mask_active'] for r in records); o = sum(r['objectness_active'] for r in records)
            totals = {'loss_bce': 0., 'loss_dice': 0., 'loss_objectness': 0., 'loss_total': 0.}
            report = {'M': m, 'O': o, 'micro_scenes': records, 'scene_count': len(window), 'losses': totals,
                      'forward_count': 0, 'status': 'NO_SUPERVISION' if m == o == 0 else 'BACKWARD_READY'}
            if m == o == 0:
                self.at_boundary = True
                return report
            self.model.train()
            for item, rec in zip(window, records):
                batch, targets = item[:2]
                prediction = self.model(batch); report['forward_count'] += 1
                result = self.criterion(prediction, targets, gt_set_exhaustive=[rec['gt_set_exhaustive']])
                if result['mask_active_scene_count'] != rec['mask_active'] or result['objectness_active_scene_count'] != rec['objectness_active']:
                    raise ValueError('Criterion/preflight active-scene count mismatch')
                for key in totals:
                    if not bool(torch.isfinite(result[key])): raise FloatingPointError(f'Nonfinite {key}')
                zero = graph_zero(prediction)
                lb = result['loss_bce']*(rec['mask_active']/m) if m else zero
                ld = result['loss_dice']*(rec['mask_active']/m) if m else zero
                lo = result['loss_objectness']*(rec['objectness_active']/o) if o else zero
                contribution = self.loss_config.bce_weight*lb + self.loss_config.dice_weight*ld + self.loss_config.objectness_weight*lo
                contribution.backward()
                values = dict(zip(totals, [lb.item(), ld.item(), lo.item(), contribution.item()]))
                rec['contribution'] = values
                for key in totals: totals[key] += values[key]
                del result, prediction, contribution, lb, ld, lo, zero
            report['gradient_groups'] = gradient_summary(self.model)
            return report
        except BaseException as exc:
            self.discard_gradients(); self.state['failed_windows'] += 1
            self.record({'status': 'FAILED_NO_STEP', 'error': repr(exc), 'state': dict(self.state)})
            raise

    def run_window(self, window, *, label):
        report = self.backward_window(window)
        if report['M'] == report['O'] == 0:
            self._consume(report, updated=False)
            report['state'] = dict(self.state); self.record(report)
            return report
        try:
            params = [p for p in self.model.parameters() if p.requires_grad]
            report['pre_clip_norm'] = joint_grad_norm(params)
            returned = torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm, error_if_nonfinite=True, foreach=False)
            report['clip_returned_norm'] = float(returned)
            gradient_summary(self.model)
            report['post_clip_norm'] = joint_grad_norm(params)
            if not math.isfinite(report['post_clip_norm']) or report['post_clip_norm'] > 1.00001:
                raise FloatingPointError('Invalid clipped norm')
            anchors = {}
            for group in self.optimizer.param_groups:
                for name, p in zip(group['param_names'], group['params']):
                    module = name.split('.')[0]
                    if module not in anchors and p.grad is not None and bool(torch.count_nonzero(p.grad)):
                        i = int(p.grad.detach().abs().reshape(-1).argmax())
                        before = p.detach().reshape(-1)[i].item()
                        anchors[module] = (p, i, {'parameter': name, 'flat_index': i, 'before': before,
                            'data_gradient_after_clip': p.grad.reshape(-1)[i].item(),
                            'weight_decay_only_delta': -group['lr']*group['weight_decay']*before})
            self.optimizer.call_context = {'purpose': PURPOSE, 'window_label': label,
                'logical_update_before': self.state['global_update_step'], 'data_cursor': self.state['data_cursor']}
            self.optimizer.step()
            report['update_anchors'] = {}
            for module, (p, i, rec) in anchors.items():
                rec['after'] = p.detach().reshape(-1)[i].item()
                rec['actual_delta'] = rec['after']-rec['before']
                rec['delta_minus_weight_decay_only'] = rec['actual_delta']-rec['weight_decay_only_delta']
                report['update_anchors'][module] = rec
            self._consume(report, updated=True)
            report['status'] = 'UPDATED'; report['state'] = dict(self.state)
            self.record(report)
            return report
        except BaseException as exc:
            self.state['failed_windows'] += 1
            self.record({'status': 'FAILED', 'error': repr(exc), 'state': dict(self.state)})
            raise
        finally:
            self.optimizer.call_context = None; self.discard_gradients()

    def _consume(self, report, *, updated):
        self.state['global_update_step'] += int(updated)
        self.state['seen_scenes'] += report['scene_count']
        self.state['data_cursor'] += report['scene_count']
        self.state['skipped_scenes'] += sum(not r['objectness_active'] and not r['mask_active'] for r in report['micro_scenes'])
        self.state['window_cursor'] += 1
