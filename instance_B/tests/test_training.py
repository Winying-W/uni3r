"""Actual-student, zero-step checks and independent concatenated-batch reference."""
import copy
import gc
import math
import unittest

import torch

from instance_B.b03_utils import tensor_fingerprints, tree_digest
from instance_B.criterion import InstanceCriterion
from instance_B.student import InstanceStudent
from instance_B.training import TrainingEngine, TrainConfig, parameter_groups, validate_parameter_groups, optimizer_group_manifest
from instance_B.tests.test_student import small_config
from instance_B.tests.test_contracts import synthetic_inputs, record_dtypes

CONTEXT = {}
EVIDENCE = {}


def make_example(kind, generator, identity, *, exhaustive=None):
    cfg = small_config()
    batch, targets, _ = synthetic_inputs(cfg.contract_spec, b=1)
    batch['images'] = torch.rand(batch['images'].shape, generator=generator)*2-1
    batch['means'] = torch.randn(batch['means'].shape, generator=generator)
    batch['geometry_confidence'] = 1+torch.rand(batch['geometry_confidence'].shape, generator=generator)*20
    batch['patch_tokens'] = {k: torch.randn(t.shape, generator=generator) for k, t in batch['patch_tokens'].items()}
    batch['metadata']['scene_id'] = [f'SYNTHETIC_B03_{identity}']
    targets['scene_id'] = list(batch['metadata']['scene_id'])
    ids = targets['instance_ids']; ids.zero_()
    if kind in ('complete', 'partial', 'zero_support'):
        ids[0, 0, 3:17, 2:13] = 7; ids[0, 1, 8:26, 12:25] = 7
        ids[0, 1, 2:8, 1:9] = 42
    ids[:, :, :2] = -1
    if kind == 'ignore': ids.fill_(-1)
    targets['valid_mask'] = ids != -1
    targets['confidence'] = targets['valid_mask'].float()*torch.linspace(.15, 1, 32)[None, None, None, :]
    if kind == 'zero_support':
        ids[0, 0, 29, 29] = 9001; targets['confidence'][0, 0, 29, 29] = 0
    record_dtypes(batch)
    flag = kind in ('complete', 'background', 'ignore', 'zero_support') if exhaustive is None else exhaustive
    return batch, targets, flag


def window(kinds, seed=73031):
    generator = torch.Generator().manual_seed(seed)
    return [make_example(kind, generator, str(i)) for i, kind in enumerate(kinds)]


def concatenate(items):
    batch = copy.deepcopy(items[0][0])
    for key in ('images', 'intrinsics', 'means', 'geometry_confidence'):
        batch[key] = torch.cat([item[0][key] for item in items])
    batch['patch_tokens'] = {k: torch.cat([item[0]['patch_tokens'][k] for item in items]) for k in batch['patch_tokens']}
    for key in ('scene_id', 'frame_ids', 'crop_resize'):
        batch['metadata'][key] = sum([copy.deepcopy(item[0]['metadata'][key]) for item in items], [])
    record_dtypes(batch)
    targets = {key: torch.cat([item[1][key] for item in items]) for key in ('instance_ids', 'valid_mask', 'confidence')}
    targets.update(scene_id=sum([item[1]['scene_id'] for item in items], []),
                   frame_ids=sum([item[1]['frame_ids'] for item in items], []), label_source='gt')
    return batch, targets, [item[2] if len(item) == 3 else False for item in items]


class TRAINING_TEST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7303)
        cls.model = InstanceStudent(small_config(), dpt_init=CONTEXT['initialization']).train()
        cls.engine = TrainingEngine(cls.model, CONTEXT['ledger'], event_log=CONTEXT['out']/'training_test_events.jsonl')
        EVIDENCE['stateful_batch_layers'] = [name for name, module in cls.model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        if EVIDENCE['stateful_batch_layers']:
            raise RuntimeError('Stateful batch-dependent layers require review before batch equivalence')

    @classmethod
    def tearDownClass(cls):
        del cls.engine, cls.model; gc.collect()

    def setUp(self):
        self.engine.discard_gradients()

    def test_01_parameter_groups(self):
        cfg = TrainConfig.from_json()
        groups = parameter_groups(self.model, cfg)
        manifest = optimizer_group_manifest(self.engine.optimizer)
        self.assertEqual(sum(x['tensor_count'] for x in manifest), 123)
        self.assertEqual([x['lr'] for x in manifest], [1e-5, 1e-4])
        for key, value in [('weight_decay', .05), ('betas', [.9, .999]), ('eps', 1e-8), ('amsgrad', False), ('foreach', False), ('fused', False)]:
            self.assertTrue(all(g[key] == value for g in manifest))
        for case in ('missing', 'duplicate', 'name_order', 'wrong_lr', 'wrong_object'):
            bad = [{**g, 'params': list(g['params']), 'param_names': list(g['param_names'])} for g in groups]
            if case == 'missing': bad[0]['params'].pop()
            if case == 'duplicate': bad[1]['params'][0] = bad[0]['params'][0]
            if case == 'name_order': bad[0]['param_names'].reverse()
            if case == 'wrong_lr': bad[0]['lr'] = 1e-4
            if case == 'wrong_object': bad[1]['params'][0] = torch.nn.Parameter(bad[1]['params'][0].detach().clone())
            with self.assertRaises(ValueError): validate_parameter_groups(self.model, bad, cfg)
        EVIDENCE['parameter_groups'] = manifest

    def compare(self, items):
        before = tensor_fingerprints(dict(self.model.state_dict()))
        accumulated = self.engine.backward_window(items)
        grads = {name: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for name, p in self.model.named_parameters()}
        self.engine.discard_gradients()
        batch, targets, flags = concatenate(items)
        prediction = self.model(batch)
        # Independent normalization reference: exactly one unchanged B-02 criterion
        # call on concatenated scenes. No training.py normalization/helper called.
        reference = InstanceCriterion(self.engine.loss_config, spec=self.model.config.contract_spec)(prediction, targets, gt_set_exhaustive=flags)
        reference['loss_total'].backward()
        errors, squared_diff, squared_norm = {}, 0., 0.
        for name, p in self.model.named_parameters():
            ref = p.grad if p.grad is not None else torch.zeros_like(p)
            torch.testing.assert_close(grads[name], ref, atol=1e-5, rtol=1e-4, msg=name)
            diff = grads[name].double()-ref.double()
            errors[name] = diff.abs().max().item()
            squared_diff += float(diff.square().sum()); squared_norm += float(ref.double().square().sum())
        scalar_errors = {}
        for key, value in accumulated['losses'].items():
            expected = reference[key].item()
            self.assertLessEqual(abs(value-expected), 1e-6+1e-5*abs(expected), key)
            scalar_errors[key] = abs(value-expected)
        self.assertEqual(before, tensor_fingerprints(dict(self.model.state_dict())))
        summary = {'M': accumulated['M'], 'O': accumulated['O'], 'losses': accumulated['losses'],
            'reference_losses': {k: reference[k].item() for k in accumulated['losses']},
            'scalar_abs_errors': scalar_errors, 'parameter_gradient_max_errors': errors,
            'gradient_max_abs_error': max(errors.values()), 'gradient_L2_difference': math.sqrt(squared_diff),
            'reference_gradient_L2_norm': math.sqrt(squared_norm), 'gradient_atol': 1e-5, 'gradient_rtol': 1e-4,
            'scalar_atol': 1e-6, 'scalar_rtol': 1e-5, 'student_mode_both_paths': 'train', 'optimizer_calls': 0}
        self.engine.discard_gradients()
        return summary

    def test_02_four_scene_accumulation_reference(self):
        summary = self.compare(window(['complete', 'partial', 'background', 'ignore']))
        self.assertEqual((summary['M'], summary['O']), (2, 3))
        EVIDENCE['accumulation_four_scene'] = summary

    def test_03_tail_window_and_background_denominator(self):
        summary = self.compare(window(['complete', 'background', 'partial'], seed=73032))
        self.assertEqual((summary['M'], summary['O']), (2, 3))
        EVIDENCE['tail_window'] = summary

    def test_04_append_ignore_not_diluted(self):
        items = window(['complete', 'ignore'], seed=73033)
        one = self.engine.backward_window(items[:1])
        gradients = {n: p.grad.clone() for n, p in self.model.named_parameters()}
        self.engine.discard_gradients()
        two = self.engine.backward_window(items)
        self.assertEqual(one['losses'], two['losses'])
        for n, p in self.model.named_parameters(): torch.testing.assert_close(gradients[n], p.grad, atol=0, rtol=0)
        self.engine.discard_gradients()
        EVIDENCE['ignore_append'] = {'status': 'PASS', 'losses': one['losses'], 'gradient_max_abs_error': 0.}

    def test_05_unsupervised_before_forward(self):
        items = window(['ignore', 'background_partial'])
        before = tree_digest({'student': dict(self.model.state_dict()), 'optimizer': self.engine.optimizer.state_dict()})
        calls = self.engine.ledger.physical_calls
        report = self.engine.run_window(items, label='TEST_NO_SUPERVISION_BEFORE_STATE')
        after = tree_digest({'student': dict(self.model.state_dict()), 'optimizer': self.engine.optimizer.state_dict()})
        self.assertEqual(before, after); self.assertEqual(calls, self.engine.ledger.physical_calls)
        self.assertEqual(report['forward_count'], 0); self.assertEqual((report['M'], report['O']), (0, 0))
        EVIDENCE['initial_no_supervision'] = report

    def test_06_contract_capacity_order_and_pseudo_errors(self):
        records = []
        for label in ('capacity', 'pseudo', 'frame_order', 'scene_order', 'nonfinite_input'):
            item = window(['complete'])[0]
            batch, targets, _ = item
            if label == 'capacity':
                targets['instance_ids'][0, 0, 10, :8] = torch.arange(1, 9)
            elif label == 'pseudo': targets['label_source'] = 'pseudo'
            elif label == 'frame_order': targets['frame_ids'][0].reverse()
            elif label == 'scene_order': targets['scene_id'][0] = 'wrong'
            else: batch['images'][0, 0, 0, 0, 0] = float('nan')
            calls = self.engine.ledger.physical_calls
            with self.assertRaises(ValueError) as caught: self.engine.run_window([item], label=label)
            self.assertEqual(self.engine.ledger.physical_calls, calls)
            self.assertTrue(all(p.grad is None for p in self.model.parameters()))
            records.append({'case': label, 'error': str(caught.exception), 'step_calls': 0})
        EVIDENCE['preflight_rejections'] = records

    def test_07_nonfinite_loss_and_gradient_no_step(self):
        original = self.engine.criterion
        class BadLoss:
            def __call__(self, *a, **kw):
                result = original(*a, **kw); result['loss_bce'] = result['loss_bce']*float('nan'); return result
        before = tree_digest(dict(self.model.state_dict()))
        calls = self.engine.ledger.physical_calls
        self.engine.criterion = BadLoss()
        try:
            with self.assertRaises(FloatingPointError): self.engine.run_window(window(['complete']), label='NONFINITE_LOSS')
        finally: self.engine.criterion = original
        hook = self.model.object_queries.register_hook(lambda g: g*float('nan'))
        try:
            with self.assertRaises(FloatingPointError): self.engine.run_window(window(['complete']), label='NONFINITE_GRADIENT')
        finally: hook.remove()
        self.assertEqual(calls, self.engine.ledger.physical_calls)
        self.assertEqual(before, tree_digest(dict(self.model.state_dict())))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        EVIDENCE['nonfinite_rejections'] = {'loss': 'PASS', 'gradient': 'PASS', 'step_calls': 0, 'gradients_cleared': True}

    def test_08_gating_and_numerical_zero_not_skip(self):
        items = window(['partial', 'zero_support'])
        records = self.engine.preflight(items)
        self.assertFalse(records[0]['negative_supervision_enabled'])
        self.assertFalse(records[1]['negative_supervision_enabled'])
        self.assertEqual(records[1]['excluded_zero_support_ids'], [9001])
        for item in items:
            p = self.model(item[0]); p['objectness_logits'].retain_grad()
            r = self.engine.criterion(p, item[1], gt_set_exhaustive=[item[2]])
            r['loss_total'].backward()
            self.assertTrue(bool((p['objectness_logits'].grad[0, r['matches'][0].unmatched_query_indices] == 0).all()))
            self.engine.discard_gradients()
        default = self.engine.preflight([items[0][:2]])[0]
        self.assertFalse(default['gt_set_exhaustive'])
        original = self.engine.criterion
        class ZeroLoss:
            def __call__(self, *a, **kw):
                r = original(*a, **kw)
                for key in ('loss_bce', 'loss_dice', 'loss_objectness', 'loss_total'): r[key] = r[key]*0
                return r
        self.engine.criterion = ZeroLoss()
        try:
            r = self.engine.backward_window(window(['complete']))
            self.assertEqual(r['status'], 'BACKWARD_READY')
            self.assertEqual((r['M'], r['O']), (1, 1)); self.assertEqual(r['losses']['loss_total'], 0)
        finally: self.engine.criterion = original; self.engine.discard_gradients()
        EVIDENCE['gating'] = {'records': records, 'default_exhaustive': False, 'partial_unmatched_direct_gradient': 0,
            'zero_loss_dispatch': 'BACKWARD_READY; fault injection of graph-connected zero, no step requested'}
