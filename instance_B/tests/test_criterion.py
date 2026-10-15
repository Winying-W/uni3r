"""Scientific loss tests with independent dense formulas and gradients."""
import copy
from dataclasses import replace
import math
import unittest

import torch
import torch.nn.functional as F

from instance_B.criterion import InstanceCriterion
from instance_B.loss_utils import LossConfig, matched_mask_components, prepare_scene
from instance_B.tests.test_matcher import case, dense, brute, match_record, SMALL

EVIDENCE = {}
KEYS = ('loss_bce', 'loss_dice', 'loss_objectness', 'loss_total')


def loss(p, t, exhaustive=None, chunk=4096):
    return InstanceCriterion(replace(LossConfig.from_json(), chunk_size=chunk), spec=SMALL)(p, t, gt_set_exhaustive=exhaustive)


def values(result):
    return {key: result[key].item() for key in KEYS}


def clone_prediction(p):
    return {key: value.detach().clone().requires_grad_(key != 'query_features') for key, value in p.items()}


def stack_cases(cases):
    ts, ps = zip(*cases)
    t = {key: torch.cat([x[key] for x in ts]) for key in ('instance_ids', 'confidence', 'valid_mask')}
    t.update(scene_id=sum([x['scene_id'] for x in ts], []), frame_ids=sum([x['frame_ids'] for x in ts], []), label_source='gt')
    p = {key: torch.cat([x[key].detach() for x in ps]).requires_grad_(key != 'query_features') for key in ps[0]}
    return t, p


class CRITERION_TEST(unittest.TestCase):
    def test_01_hand_computed(self):
        _, t, p = case((7, 0), [1, .5], positions=[0, 256])
        with torch.no_grad():
            p['mask_logits'].zero_()
            p['mask_logits'][0, 0].flatten()[0] = math.log(3)
            p['mask_logits'][0, 1].flatten()[0] = -math.log(3)
            p['objectness_logits'][0] = torch.tensor([math.log(3), 0], dtype=torch.float64)
        r = loss(p, t, [True])
        expected = dict(zip(KEYS, [.4228371084878356, .24999981250014058, .1784983952538877, 3.7211813954476565]))
        self.assertEqual(r['matches'][0].matched_query_indices, [0])
        for key in KEYS: self.assertAlmostEqual(r[key].item(), expected[key], places=12)
        EVIDENCE['hand_computed'] = {'expected': expected, 'actual': values(r),
            'absolute_error': {key: abs(r[key].item()-expected[key]) for key in KEYS},
            'matching': match_record(r['matches'][0])}

    def test_02_empty_exhaustive_background(self):
        _, t, p = case((0, 0), [1, .5])
        with torch.no_grad(): p['objectness_logits'].zero_()
        r = loss(p, t, [True])
        self.assertAlmostEqual(r['loss_objectness'].item(), .1*math.log(2), places=14)
        self.assertAlmostEqual(r['loss_total'].item(), .2*math.log(2), places=14)
        self.assertEqual(r['mask_active_scene_count'], 0)
        r['loss_total'].backward()
        self.assertTrue(bool((p['mask_logits'].grad == 0).all()))
        torch.testing.assert_close(p['objectness_logits'].grad, torch.full_like(p['objectness_logits'], .05), atol=1e-14, rtol=0)
        EVIDENCE['empty_background'] = values(r)

    def test_03_confidence_relative_once(self):
        _, t, p = case((7, 0, 7), [.7, .4, .1])
        r = loss(p, t, [True])
        scaled = copy.deepcopy(t); scaled['confidence'] *= 1e-12
        rr = loss(p, scaled, [True])
        for key in KEYS: torch.testing.assert_close(r[key], rr[key], atol=1e-12, rtol=1e-10)
        changed = copy.deepcopy(t); changed['confidence'].flatten()[1] = .03
        actual = loss(p, changed, [True])
        _, bc, dc, _, cost = dense(p, changed)
        _, qi, _ = brute(cost.detach()); ti = list(range(len(qi)))
        torch.testing.assert_close(actual['loss_bce'], bc[list(qi), ti].mean(), atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(actual['loss_dice'], dc[list(qi), ti].mean(), atol=1e-12, rtol=1e-10)
        self.assertNotAlmostEqual(r['loss_bce'].item(), actual['loss_bce'].item(), places=6)

    def test_04_ignore_invalid_zero_confidence_gradients(self):
        _, t, p = case((7, 0, 42, 0, -1), [1, .3, 0, 0, 0])
        t['valid_mask'].flatten()[3] = False
        r = loss(p, t, [True]); r['loss_total'].backward()
        zero = t['confidence'][0] == 0
        self.assertTrue(bool((p['mask_logits'].grad[0, :, zero] == 0).all()))
        matched = r['matches'][0].matched_query_indices[0]
        self.assertGreater(p['mask_logits'].grad[0, matched].flatten()[1].item(), 0)
        pp = clone_prediction(p)
        with torch.no_grad(): pp['mask_logits'][0, :, zero] = 1000
        rr = loss(pp, t, [True])
        for key in KEYS: torch.testing.assert_close(r[key], rr[key], atol=0, rtol=0)
        torch.testing.assert_close(r['matches'][0].total_cost, rr['matches'][0].total_cost, atol=0, rtol=0)

    def test_05_cross_view_id_switch_penalty(self):
        _, t, p = case((7, 42, 7, 42), positions=[0, 1, 256, 257])
        with torch.no_grad():
            p['mask_logits'].fill_(-8)
            p['mask_logits'][0, 0, :, 0, 0] = 8
            p['mask_logits'][0, 1, :, 0, 1] = 8
            p['objectness_logits'].zero_()
        good = loss(p, t, [True])
        pp = clone_prediction(p)
        with torch.no_grad(): pp['mask_logits'][0, :, 1] = p['mask_logits'][0, [1, 0], 1]
        bad = loss(pp, t, [True])
        swapped = copy.deepcopy(t)
        swapped['instance_ids'][0, 1, 0, :2] = torch.tensor([42, 7])
        bad_gt = loss(p, swapped, [True])
        self.assertGreater(bad['loss_total'].item(), good['loss_total'].item()+1)
        self.assertGreater(bad_gt['loss_total'].item(), good['loss_total'].item()+1)
        EVIDENCE['cross_view_switch'] = {'correct': values(good), 'prediction_switch': values(bad), 'GT_second_view_only_switch': values(bad_gt)}

    def test_06_global_id_and_view_permutation_invariance(self):
        _, t, p = case((7, 42, 9001, 7, 0), [.3, 1, .9, .2, 1], q=4, positions=[0, 256, 257, 1, 258])
        r = loss(p, t, [True])
        tt = copy.deepcopy(t)
        for old, new in {7: 100, 42: 3, 9001: 51}.items(): tt['instance_ids'][t['instance_ids'] == old] = new
        renamed = loss(p, tt, [True])
        tt = copy.deepcopy(t)
        for key in ('instance_ids', 'valid_mask', 'confidence'): tt[key] = tt[key][:, [1, 0]]
        tt['frame_ids'] = [list(reversed(row)) for row in tt['frame_ids']]
        pp = dict(p); pp['mask_logits'] = p['mask_logits'][:, :, [1, 0]]
        viewed = loss(pp, tt, [True])
        pp = {key: v[:, [3, 1, 0, 2]] for key, v in p.items()}
        queried = loss(pp, t, [True])
        for rr in (renamed, viewed, queried):
            for key in KEYS: torch.testing.assert_close(r[key], rr[key], atol=1e-10, rtol=1e-8)

    def test_07_exhaustiveness_branches(self):
        records = []
        variants = [('all_ignore', [-1], [0], True, 0, 0),
                    ('all_zero_confidence', [7, 0], [0, 0], True, 0, 0),
                    ('background_complete', [0], [1], True, 0, 1),
                    ('background_partial', [0], [1], False, 0, 0),
                    ('zero_support_positive', [42, 0], [0, 1], True, 0, 0),
                    ('positive_partial', [7, 0], [1, 1], False, 1, 1),
                    ('positive_complete', [7, 0], [1, 1], True, 1, 1),
                    ('positive_plus_zero_support', [7, 42, 0], [1, 0, 1], True, 1, 1)]
        for name, ids, conf, e, ma, oa in variants:
            with self.subTest(name=name):
                _, t, p = case(ids, conf, q=3)
                r = loss(p, t, [e]); r['loss_total'].backward()
                self.assertEqual(r['mask_active_scene_count'], ma)
                self.assertEqual(r['objectness_active_scene_count'], oa)
                self.assertTrue(bool(torch.isfinite(p['mask_logits'].grad).all()))
                self.assertTrue(bool(torch.isfinite(p['objectness_logits'].grad).all()))
                unmatched = r['matches'][0].unmatched_query_indices
                if not r['scenes'][0]['negative_supervision_enabled']:
                    self.assertTrue(bool((p['objectness_logits'].grad[0, unmatched] == 0).all()))
                if not oa:
                    self.assertEqual(r['loss_total'].item(), 0)
                    self.assertTrue(bool((p['mask_logits'].grad == 0).all() and (p['objectness_logits'].grad == 0).all()))
                if 'zero_support' in name or name == 'all_zero_confidence':
                    self.assertIn('NEGATIVE_SUPERVISION_DISABLED_ZERO_SUPPORT_INSTANCE', r['scenes'][0]['reasons'])
                records.append({'case': name, 'loss': values(r), 'scenes': r['scenes']})
        EVIDENCE['exhaustiveness_cases'] = records

    def test_08_default_partial_and_bool_validation(self):
        _, t, p = case((0,))
        self.assertEqual(loss(p, t)['loss_total'].item(), 0)
        for flag in (True, [1], [], [False, True], 'true'):
            with self.assertRaisesRegex(ValueError, 'gt_set_exhaustive'): loss(p, t, flag)

    def test_09_batch_isolation_and_permutation(self):
        _, t1, p1 = case((7, 0), q=3)
        _, t2, p2 = case((0, 7), q=3)
        t, p = stack_cases([(t1, p1), (t2, p2)])
        r = loss(p, t, [True, False])
        pp = clone_prediction(p)
        with torch.no_grad(): pp['mask_logits'][1].mul_(-5); pp['objectness_logits'][1].add_(3)
        rr = loss(pp, t, [True, False])
        self.assertEqual(r['scenes'][0], rr['scenes'][0])
        self.assertEqual(r['matches'][0].matched_query_indices, rr['matches'][0].matched_query_indices)
        self.assertEqual([m.batch_index for m in r['matches']], [0, 1])
        # Same scene_id strings deliberately verify that aggregation is by batch item.
        t_rev, p_rev = stack_cases([(t2, p2), (t1, p1)])
        rev = loss(p_rev, t_rev, [False, True])
        for key in KEYS: torch.testing.assert_close(r[key], rev[key], atol=1e-12, rtol=1e-10)

    def test_10_ignore_append_and_duplicate_do_not_dilute(self):
        _, t, p = case((7, 0), q=3)
        one = loss(p, t, [True]); one['loss_total'].backward()
        _, ignore, pi = case((-1,), [0], q=3)
        tt, pp = stack_cases([(t, p), (ignore, pi)])
        appended = loss(pp, tt, [True, True]); appended['loss_total'].backward()
        for key in KEYS: torch.testing.assert_close(one[key], appended[key], atol=0, rtol=0)
        for key in ('mask_logits', 'objectness_logits'):
            torch.testing.assert_close(p[key].grad[0], pp[key].grad[0], atol=0, rtol=0)
            self.assertTrue(bool((pp[key].grad[1] == 0).all()))
        tt, pp = stack_cases([(t, p), (t, p)])
        duplicate = loss(pp, tt, [True, True])
        for key in KEYS: torch.testing.assert_close(one[key], duplicate[key], atol=0, rtol=0)
        _, background, pb = case((0,), q=3)
        tt, pp = stack_cases([(t, p), (background, pb)])
        bg = loss(pp, tt, [True, True])
        for key in ('loss_bce', 'loss_dice'): torch.testing.assert_close(one[key], bg[key], atol=0, rtol=0)
        self.assertEqual(bg['mask_active_scene_count'], 1)
        self.assertEqual(bg['objectness_active_scene_count'], 2)
        EVIDENCE['batch_normalization'] = {'one': values(one), 'ignore_appended': values(appended), 'duplicated': values(duplicate), 'background_appended': values(bg), 'original_gradient_max_difference': 0}

    def test_11_extreme_logits_and_tiny_positive_confidence(self):
        records = []
        for value in (-1000, -100, 0, 100, 1000):
            _, t, p = case((7, 0), [1e-300, 5e-301])
            with torch.no_grad(): p['mask_logits'].fill_(value); p['objectness_logits'].fill_(value)
            r = loss(p, t, [True]); r['loss_total'].backward()
            self.assertEqual(r['matches'][0].target_instance_ids, [7])
            for key in ('mask_logits', 'objectness_logits'): self.assertTrue(bool(torch.isfinite(p[key].grad).all()))
            self.assertTrue(all(math.isfinite(r[key].item()) for key in KEYS))
            records.append({'logit': value, 'S': r['scenes'][0]['S'], 'loss': values(r)})
        _, t, p = case((-1,), [0])
        with torch.no_grad(): p['mask_logits'].fill_(1e308); p['objectness_logits'].fill_(-1e308)
        r = loss(p, t, [True]); r['loss_total'].backward()
        self.assertEqual(r['loss_total'].item(), 0)
        self.assertTrue(bool((p['mask_logits'].grad == 0).all()))
        EVIDENCE['extreme_logits'] = records

    def test_12_dense_loss_and_gradient_parity(self):
        records = []
        for dtype in (torch.float64, torch.float32):
            for chunk in (1, 7, 4096):
                _, t, p = case((7, 42, 0, 7, -1, 9001, 0, 42), [1, .2, .4, .7, 0, .9, .3, .1], q=4, dtype=dtype)
                refp = clone_prediction(p)
                r = loss(p, t, [True], chunk=chunk)
                ids, bc, dc, _, cost = dense(refp, t)
                _, qi, _ = brute(cost.detach()); ti = list(range(len(ids)))
                lb, ld = bc[list(qi), ti].mean(), dc[list(qi), ti].mean()
                o = refp['objectness_logits'][0].double()
                unmatched = [j for j in range(4) if j not in qi]
                lo = (F.softplus(-o[list(qi)]).sum()+.1*F.softplus(o[unmatched]).sum())/4
                total = 5*lb+5*ld+2*lo
                expected = dict(zip(KEYS, (lb, ld, lo, total)))
                for key in KEYS: torch.testing.assert_close(r[key], expected[key], atol=1e-10, rtol=1e-8)
                r['loss_total'].backward(); total.backward()
                errors = {}
                for key in ('mask_logits', 'objectness_logits'):
                    atol, rtol = (1e-10, 1e-8) if dtype == torch.float64 else (1e-6, 1e-5)
                    torch.testing.assert_close(p[key].grad, refp[key].grad, atol=atol, rtol=rtol)
                    errors[key] = (p[key].grad-refp[key].grad).abs().max().item()
                records.append({'dtype': str(dtype), 'chunk': chunk, 'actual': values(r),
                                'expected': {key: v.item() for key, v in expected.items()}, 'gradient_max_errors': errors})
        EVIDENCE['dense_loss_gradient_parity'] = records

    def test_13_fixed_assignment_gradcheck(self):
        _, t, p = case((7, 0, 7), [1, .5, .2], q=2)
        scene = prepare_scene(t, 0)
        cfg = replace(LossConfig.from_json(), chunk_size=1)
        # Kernel-only test uses tiny spatial size; public fixtures still use SMALL.
        scene.pixel_indices = torch.arange(3)
        z = torch.tensor([[1.1, -.7, .8]], dtype=torch.float64, requires_grad=True)
        def kernel(logits):
            bc, dc = matched_mask_components(logits, scene, [0], [0], cfg)
            return 5*bc.mean()+5*dc.mean()
        self.assertTrue(torch.autograd.gradcheck(kernel, (z,), eps=1e-6, atol=1e-5, rtol=1e-3))
        EVIDENCE['fixed_assignment_gradcheck'] = {'status': 'PASS', 'kernel_only': True, 'discrete_solver_differentiated': False}

    def test_14_single_view_visibility_negative_other_view(self):
        _, t, p = case((7, 0), [1, 1], positions=[0, 256])
        r = loss(p, t, [True]); r['loss_total'].backward()
        q = r['matches'][0].matched_query_indices[0]
        self.assertGreater(p['mask_logits'].grad[0, q, 1, 0, 0].item(), 0)
        tt = copy.deepcopy(t)
        tt['instance_ids'][:, 1] = -1; tt['valid_mask'][:, 1] = False; tt['confidence'][:, 1] = 0
        pp = clone_prediction(p); rr = loss(pp, tt, [True]); rr['loss_total'].backward()
        self.assertTrue(bool((pp['mask_logits'].grad[:, :, 1] == 0).all()))
        self.assertEqual(rr['matches'][0].target_instance_ids, [7])


if __name__ == '__main__':
    unittest.main(verbosity=2)
