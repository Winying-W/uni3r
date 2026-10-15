"""Malformed payloads use shallow tensor-sharing copies; no extra optimizer steps."""
import copy
import io
import json
import unittest

import torch

from instance_B.b03_utils import tree_digest, write_json
from instance_B.checkpointing import validate_payload, save_checkpoint, load_verified_payload, restore_checkpoint

CONTEXT = {}
EVIDENCE = {}


class CHECKPOINT_TEST(unittest.TestCase):
    def test_01_payload_complete_and_purpose_isolated(self):
        c = CONTEXT
        self.assertTrue(validate_payload(c['payload'], c['engine'], c['samples'], c['generators']))
        self.assertEqual(c['payload']['purpose'], 'SYNTHETIC_ENGINE_DIAGNOSTIC_NOT_FOR_REAL_TRAINING')
        self.assertEqual(c['payload']['counters']['global_update_step'], 1)
        EVIDENCE['payload_valid'] = True

    def test_02_invalid_payloads_do_not_mutate_live_engine(self):
        c = CONTEXT; engine = c['engine']; original = c['payload']
        before = tree_digest({'model': dict(engine.model.state_dict()), 'optimizer': engine.optimizer.state_dict(), 'state': engine.state})
        rng = torch.get_rng_state().clone()
        gen = {k: g.get_state().clone() for k, g in c['generators'].items()}
        records = []
        cases = ['Q', 'layers', 'contract', 'base', 'DPT', 'source', 'loss', 'group_order', 'optimizer_name_order',
                 'student_missing_key', 'student_shape', 'student_dtype', 'student_nonfinite', 'purpose', 'format',
                 'optimizer_shape', 'optimizer_nonfinite', 'negative_second_moment', 'local_step', 'missing_cursor',
                 'missing_rng', 'generator_names', 'mid_window', 'backbone_key']
        for name in cases:
            bad = dict(original)
            if name in ('Q', 'layers', 'loss'):
                bad['configs'] = copy.deepcopy(original['configs'])
                if name == 'Q': bad['configs']['student']['num_queries'] = 8
                if name == 'layers': bad['configs']['student']['decoder_layers'] = 3
                if name == 'loss': bad['configs']['loss']['bce_weight'] = 6
            elif name in ('contract', 'base', 'DPT', 'source'):
                bad['provenance'] = copy.deepcopy(original['provenance'])
                key = {'contract': 'contract_sha256', 'base': 'base_checkpoint_sha256', 'DPT': 'dpt_initialization_sha256'}
                if name == 'source': bad['provenance']['source_sha256']['instance_B/student.py'] = '0'*64
                else: bad['provenance'][key[name]] = '0'*64
            elif name == 'group_order':
                bad['optimizer_groups'] = copy.deepcopy(original['optimizer_groups']); bad['optimizer_groups'][0]['param_names'].reverse()
            elif name.startswith('student_') or name == 'backbone_key':
                bad['student_state_dict'] = dict(original['student_state_dict'])
                key = 'objectness_head.bias'
                if name == 'student_missing_key': bad['student_state_dict'].pop(key)
                elif name == 'student_shape': bad['student_state_dict'][key] = torch.zeros(2)
                elif name == 'student_dtype': bad['student_state_dict'][key] = original['student_state_dict'][key].double()
                elif name == 'student_nonfinite': bad['student_state_dict'][key] = torch.full_like(original['student_state_dict'][key], float('nan'))
                else: bad['student_state_dict']['vggt.backbone.forbidden'] = torch.zeros(1)
            elif name == 'purpose': bad['purpose'] = 'REAL_GT_TRAINING'
            elif name == 'format': bad['format_version'] = 999
            elif name in ('optimizer_name_order', 'optimizer_shape', 'optimizer_nonfinite', 'negative_second_moment', 'local_step'):
                opt = original['optimizer_state_dict']
                bad['optimizer_state_dict'] = dict(opt)
                if name == 'optimizer_name_order':
                    bad['optimizer_state_dict']['param_groups'] = copy.deepcopy(opt['param_groups'])
                    bad['optimizer_state_dict']['param_groups'][0]['param_names'].reverse()
                else:
                    states = dict(opt['state']); index = min(states, key=lambda i: states[i]['exp_avg'].numel())
                    states[index] = dict(states[index]); bad['optimizer_state_dict']['state'] = states
                    if name == 'optimizer_shape': states[index]['exp_avg'] = torch.zeros(99)
                    if name == 'optimizer_nonfinite': states[index]['exp_avg'] = torch.full_like(states[index]['exp_avg'], float('inf'))
                    if name == 'negative_second_moment': states[index]['exp_avg_sq'] = -torch.ones_like(states[index]['exp_avg_sq'])
                    if name == 'local_step': states[index]['step'] = torch.tensor(100.)
            elif name == 'missing_cursor': bad['counters'] = {k: v for k, v in original['counters'].items() if k != 'data_cursor'}
            elif name == 'missing_rng': bad.pop('rng')
            elif name == 'generator_names':
                bad['rng'] = dict(original['rng']); bad['rng']['generators'] = {}
            elif name == 'mid_window': bad['window_boundary'] = False
            with self.subTest(case=name):
                with self.assertRaises((ValueError, TypeError)) as caught:
                    validate_payload(bad, engine, c['samples'], c['generators'])
                records.append({'case': name, 'status': 'REJECTED', 'error': str(caught.exception)})
        self.assertEqual(before, tree_digest({'model': dict(engine.model.state_dict()), 'optimizer': engine.optimizer.state_dict(), 'state': engine.state}))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, g in c['generators'].items(): self.assertTrue(torch.equal(gen[k], g.get_state()))
        EVIDENCE['invalid_payloads'] = records
        EVIDENCE['invalid_load_live_state_unchanged'] = True

    def test_03_truncated_and_bad_receipt(self):
        c = CONTEXT; out = c['out']
        corrupt = out/'truncated_diagnostic.pt'
        with corrupt.open('xb') as f: f.write(b'PK\x03\x04TRUNCATED')
        from instance_B.b03_utils import file_sha256
        receipt = dict(c['receipt']); receipt.update(path=str(corrupt), bytes=corrupt.stat().st_size, sha256=file_sha256(corrupt))
        corrupt_receipt = out/'truncated_receipt.json'; write_json(corrupt_receipt, receipt)
        with self.assertRaises(Exception): load_verified_payload(corrupt, corrupt_receipt)
        wrong = dict(c['receipt']); wrong['sha256'] = '0'*64
        bad_path = out/'bad_hash_receipt.json'; write_json(bad_path, wrong)
        before = tree_digest({'student': dict(c['engine'].model.state_dict()), 'optimizer': c['engine'].optimizer.state_dict()})
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            restore_checkpoint(c['path'], bad_path, c['engine'], c['initialization'], c['samples'], c['generators'])
        self.assertEqual(before, tree_digest({'student': dict(c['engine'].model.state_dict()), 'optimizer': c['engine'].optimizer.state_dict()}))
        EVIDENCE['truncated_file'] = 'REJECTED with weights_only=True; no unsafe fallback'
        EVIDENCE['bad_receipt_no_partial_restore'] = 'PASS'

    def test_04_no_overwrite_and_boundary_only(self):
        c = CONTEXT
        with self.assertRaises(FileExistsError): save_checkpoint(c['engine'], c['path'], c['samples'], c['generators'])
        c['engine'].at_boundary = False
        try:
            with self.assertRaisesRegex(ValueError, 'window boundary'):
                save_checkpoint(c['engine'], c['out']/'forbidden_midwindow.pt', c['samples'], c['generators'])
        finally: c['engine'].at_boundary = True
        self.assertFalse((c['out']/'forbidden_midwindow.pt').exists())
        EVIDENCE['save_boundaries'] = {'existing_file': 'REJECTED', 'mid_window': 'REJECTED_BEFORE_WRITE'}
