"""Independent tiny references and public-contract synthetic matching checks."""
import copy
from dataclasses import replace
import itertools
import math
import unittest

import torch
import torch.nn.functional as F

from instance_B.matcher import SceneMatcher, solve_assignment, CapacityError
from instance_B.loss_utils import LossConfig
from instance_B.tests.test_contracts import SMALL, synthetic_inputs

EVIDENCE = {}


def case(labels=(7, 0), confidence=None, q=2, dtype=torch.float64, positions=None):
    batch, targets, prediction = synthetic_inputs(SMALL, q=q)
    targets['instance_ids'].fill_(-1)
    targets['valid_mask'].fill_(False)
    targets['confidence'] = torch.zeros_like(targets['confidence'], dtype=dtype)
    positions = list(range(len(labels))) if positions is None else positions
    for i, label in enumerate(labels):
        at = positions[i]
        targets['instance_ids'].view(-1)[at] = label
        targets['valid_mask'].view(-1)[at] = label != -1
        targets['confidence'].view(-1)[at] = (1 if label != -1 else 0) if confidence is None else confidence[i]
    generator = torch.Generator().manual_seed(7302)
    prediction = {k: torch.randn(v.shape, generator=generator, dtype=dtype) for k, v in prediction.items()}
    for key in ('mask_logits', 'objectness_logits'):
        prediction[key].requires_grad_()
    return batch, targets, prediction


def brute(cost):
    q, k = cost.shape
    choices = list(itertools.permutations(range(q), k))
    values = [sum(float(cost[rows[j], j]) for j in range(k)) for rows in choices]
    best = min(values)
    return best, choices[values.index(best)], sum(abs(x-best) < 1e-10 for x in values)


def dense(prediction, targets, b=0):
    """Independent QxKxN reference; never call optimized reduction helpers."""
    z = prediction['mask_logits'][b].flatten(1).double()
    o = prediction['objectness_logits'][b].double()
    labels = targets['instance_ids'][b].flatten()
    w = targets['valid_mask'][b].flatten().double()*targets['confidence'][b].flatten().double()
    ids = sorted(set(labels[(w > 0) & (labels > 0)].tolist()))
    u = w/w.sum()
    t = torch.stack([(labels == i).double() for i in ids])
    bce = (u*(t[None]*F.softplus(-z[:, None])+(1-t[None])*F.softplus(z[:, None]))).sum(-1)
    p = z.sigmoid()
    dice = 1-(2*(u*p[:, None]*t[None]).sum(-1)+1e-6)/((u*p).sum(-1)[:, None]+(u*t).sum(-1)[None]+1e-6)
    obj = F.softplus(-o)
    return ids, bce, dice, obj, 5*bce+5*dice+2*obj[:, None]


def match_record(m):
    return {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in vars(m).items()}


class MATCHER_TEST(unittest.TestCase):
    def test_01_global_optimum_not_greedy(self):
        cost = torch.tensor([[1., 2.], [2., 100.]])
        qi, ti = solve_assignment(cost)
        self.assertEqual(dict(zip(qi, ti)), {0: 1, 1: 0})
        self.assertEqual(cost[qi, ti].sum().item(), 4)
        EVIDENCE['nongreedy'] = {'cost': cost.tolist(), 'query': qi, 'target': ti, 'value': 4}

    def test_02_rectangular_exhaustive(self):
        generator = torch.Generator().manual_seed(7302)
        records = []
        for q in range(1, 6):
            for k in range(q+1):
                for ties in (False, True):
                    cost = torch.zeros(q, k, dtype=torch.float64) if ties else torch.randn(q, k, generator=generator, dtype=torch.float64)-1
                    expected, _, optima = brute(cost)
                    qi, ti = solve_assignment(cost)
                    actual = cost[qi, ti].sum().item()
                    self.assertAlmostEqual(actual, expected, places=12)
                    self.assertEqual(sorted(ti), list(range(k)))
                    self.assertEqual(len(set(qi)), k)
                    records.append({'Q': q, 'K': k, 'ties': ties, 'optima': optima, 'actual': actual, 'expected': expected})
        EVIDENCE['exhaustive_solver'] = records

    def test_03_dense_matrices_and_chunks(self):
        records = []
        for dtype in (torch.float64, torch.float32):
            _, t, p = case((7, 42, 9001, 0, 7, 0, -1, 42, 0), [1, .2, .7, .1, .3, .9, 0, 0, .001], q=4, dtype=dtype)
            ids, bce, dice, obj, cost = dense(p, t)
            expected, _, _ = brute(cost.detach())
            for chunk in (1, 7, 4096):
                m = SceneMatcher(replace(LossConfig.from_json(), chunk_size=chunk), spec=SMALL)(p, t)[0]
                self.assertEqual(m.target_instance_ids, ids)
                errors = {}
                for name, a, b in [('BCE', m.bce_cost, bce), ('Dice', m.dice_cost, dice), ('objectness', m.objectness_cost, obj), ('cost', m.total_cost, cost)]:
                    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
                    errors[name] = (a-b).abs().max().item()
                self.assertAlmostEqual(m.optimal_cost, expected, places=10)
                self.assertTrue(p['mask_logits'].requires_grad)
                self.assertFalse(m.total_cost.requires_grad)
                records.append({'dtype': str(dtype), 'chunk': chunk, 'max_errors': errors, 'optimal_value': expected})
        EVIDENCE['dense_matrix_parity'] = records

    def test_04_unique_id_and_query_equivariance(self):
        _, t, p = case((7, 42, 9001, 7, 42, 9001, 0), q=4)
        with torch.no_grad():
            p['mask_logits'].fill_(-8)
            for q, label in enumerate((7, 42, 9001)):
                p['mask_logits'][0, q][t['instance_ids'][0] == label] = 8
            p['objectness_logits'].fill_(2)
        original = SceneMatcher(spec=SMALL)(p, t)[0]
        self.assertEqual(brute(original.total_cost)[2], 1)
        rename = {7: 900, 42: 5, 9001: 77}
        tt = copy.deepcopy(t)
        for old, new in rename.items():
            tt['instance_ids'][t['instance_ids'] == old] = new
        renamed = SceneMatcher(spec=SMALL)(p, tt)[0]
        self.assertEqual(dict(zip(renamed.matched_query_indices, renamed.matched_instance_ids)),
                         {q: rename[i] for q, i in zip(original.matched_query_indices, original.matched_instance_ids)})
        perm = [3, 2, 0, 1]
        pp = {key: value[:, perm] for key, value in p.items()}
        changed = SceneMatcher(spec=SMALL)(pp, t)[0]
        self.assertEqual({perm[q]: i for q, i in zip(changed.matched_query_indices, changed.matched_instance_ids)},
                         dict(zip(original.matched_query_indices, original.matched_instance_ids)))
        EVIDENCE['unique_equivariance'] = {'original': match_record(original), 'renamed': match_record(renamed), 'query_permuted': match_record(changed)}

    def test_05_single_view_object_and_zero_support(self):
        _, t, p = case((7, 0, 42, 9001), [1, 1, 0, 0], positions=[0, 256, 2, 257], q=2)
        m = SceneMatcher(spec=SMALL)(p, t)[0]
        self.assertEqual(m.target_instance_ids, [7])
        self.assertEqual(m.excluded_zero_support_ids, [42, 9001])
        self.assertEqual(m.supervised_pixel_count, 2)
        self.assertEqual(m.matched_instance_ids, [7])

    def test_06_capacity_context_and_k_equals_q(self):
        _, t, p = case((7, 42), q=2)
        self.assertEqual(len(SceneMatcher(spec=SMALL)(p, t)[0].matched_query_indices), 2)
        _, t, p = case((7, 42, 9001), q=2)
        with self.assertRaisesRegex(CapacityError, 'batch_index=0.*scene_id=.*K=3 Q=2'):
            SceneMatcher(spec=SMALL)(p, t)

    def test_07_label_source_rejection(self):
        records = []
        for source in ('pseudo', ['gt', 'pseudo'], 'mixed', 'unknown', None):
            _, t, p = case()
            t['label_source'] = source
            with self.assertRaisesRegex(ValueError, 'label_source') as caught:
                SceneMatcher(spec=SMALL)(p, t)
            records.append({'source': source, 'error': str(caught.exception)})
        EVIDENCE['source_rejections'] = records

    def test_08_invalid_predictions_and_targets(self):
        changes = [lambda t,p: p.update(mask_logits=p['mask_logits'][:, :0]),
                   lambda t,p: p.update(objectness_logits=torch.zeros(1, 3)),
                   lambda t,p: p.update(mask_logits=p['mask_logits'].long()),
                   lambda t,p: t.update(instance_ids=t['instance_ids'].int()),
                   lambda t,p: t.update(frame_ids=[['same', 'same']]),
                   lambda t,p: t.update(scene_id=[]),
                   lambda t,p: t['instance_ids'].fill_(-2),
                   lambda t,p: t['confidence'].fill_(2),
                   lambda t,p: t['confidence'].requires_grad_(),
                   lambda t,p: t['valid_mask'].fill_(True)]
        for change in changes:
            with self.subTest(change=repr(change)):
                _, t, p = case()
                change(t, p)
                with self.assertRaises(ValueError): SceneMatcher(spec=SMALL)(p, t)
        for key in ('mask_logits', 'objectness_logits', 'query_features'):
            for nonfinite in (float('nan'), float('inf'), -float('inf')):
                _, t, p = case()
                with torch.no_grad(): p[key].reshape(-1)[-1] = nonfinite
                with self.assertRaisesRegex(ValueError, 'NaN/Inf'): SceneMatcher(spec=SMALL)(p, t)

    def test_09_empty_solver_and_match(self):
        for labels, conf in [((-1,), [0]), ((0,), [1]), ((7,), [0])]:
            _, t, p = case(labels, conf)
            m = SceneMatcher(spec=SMALL)(p, t)[0]
            self.assertEqual(m.matched_query_indices, [])
            self.assertEqual(m.unmatched_query_indices, [0, 1])
        with self.assertRaises(ValueError): solve_assignment(torch.tensor([[float('nan')]]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
