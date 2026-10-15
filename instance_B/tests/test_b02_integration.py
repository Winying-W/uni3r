"""B-02 loss through the unchanged real B-01 student, using synthetic inputs."""
import copy
import hashlib
import math
import unittest

import torch

from instance_B.criterion import InstanceCriterion
from instance_B.student import InstanceStudent
from instance_B.tests import test_student
from instance_B.tests.test_contracts import synthetic_inputs
from uni3r.instance.contracts import validate_training_batch, ContractError

EVIDENCE = {}


def tensor_digest(tensors):
    sha = hashlib.sha256()
    for key, tensor in tensors:
        sha.update(key.encode())
        sha.update(tensor.detach().contiguous().numpy().tobytes())
    return sha.hexdigest()


class B02_INTEGRATION_TEST(unittest.TestCase):
    def test_01_student_small_backward(self):
        torch.manual_seed(7302)
        cfg = test_student.small_config()
        init = test_student.load_init()
        model = InstanceStudent(cfg, dpt_init=init).train()
        batch = test_student.synthetic_scene(cfg, b=2, seed=7302)
        _, targets, _ = synthetic_inputs(cfg.contract_spec, b=2)
        ids = targets['instance_ids']; ids.zero_()
        ids[0, 0, 2:13, 3:15] = 7; ids[0, 1, 4:15, 7:18] = 7
        ids[0, 1, 20:27, 2:11] = 42
        ids[1, 0, 7:23, 1:8] = 7; ids[1, 1, 1:9, 19:30] = 7
        ids[1, 0, 20:30, 16:28] = 9001
        ids[:, :, :2, :] = -1
        targets['valid_mask'] = ids != -1
        ramp = torch.linspace(.2, 1., cfg.width).reshape(1, 1, 1, -1)
        targets['confidence'] = targets['valid_mask'].float()*ramp
        flags = [True, False]
        before = tensor_digest(model.named_parameters())
        reference_before = tensor_digest(sorted(init.state.items()))
        prediction = model(batch)
        validate_training_batch(batch, targets, prediction, spec=cfg.contract_spec)
        criterion = InstanceCriterion(spec=cfg.contract_spec)
        result = criterion(prediction, targets, gt_set_exhaustive=flags)
        result['loss_total'].backward()
        groups, parameters = {}, []
        for name, parameter in model.named_parameters():
            group = name.split('.')[0]
            entry = groups.setdefault(group, {'parameter_tensors': 0, 'with_grad': 0, 'finite_tensors': 0,
                                              'nonzero_tensors': 0, 'nonzero_elements': 0, 'squared_norm': 0.})
            entry['parameter_tensors'] += 1
            grad = parameter.grad
            self.assertIsNotNone(grad, name)
            finite = bool(torch.isfinite(grad).all())
            nonzero = int(torch.count_nonzero(grad))
            norm = torch.linalg.vector_norm(grad.double()).item()
            entry['with_grad'] += 1; entry['finite_tensors'] += int(finite)
            entry['nonzero_tensors'] += int(nonzero > 0); entry['nonzero_elements'] += nonzero
            entry['squared_norm'] += norm**2
            parameters.append({'name': name, 'finite': finite, 'nonzero_elements': nonzero, 'grad_norm': norm})
        for entry in groups.values(): entry['grad_norm'] = math.sqrt(entry.pop('squared_norm'))
        required = {'instance_dpt', 'adapter', 'position_encoder', 'object_queries', 'decoder',
                    'mask_features', 'query_mask_embedding', 'objectness_head'}
        self.assertTrue(required <= set(groups))
        frozen = [batch['images'], batch['intrinsics'], batch['means'], batch['geometry_confidence'], *batch['patch_tokens'].values()]
        after = tensor_digest(model.named_parameters())
        reference_after = tensor_digest(sorted(init.state.items()))
        evidence = {'status': 'PASS', 'evidence_kind': 'SYNTHETIC', 'student_config': cfg.to_dict(),
            'gt_set_exhaustive': flags, 'shapes': {k: list(v.shape) for k, v in prediction.items()},
            'output_finite': {k: bool(torch.isfinite(v).all()) for k, v in prediction.items()},
            'prediction_dtype': str(prediction['mask_logits'].dtype), 'loss_dtype': result['scalar_dtype'],
            'losses': {key: result[key].item() for key in ('loss_total', 'loss_bce', 'loss_dice', 'loss_objectness')},
            'matches': [{'batch_index': m.batch_index, 'scene_id': m.scene_id, 'query': m.matched_query_indices,
                         'original_instance_id': m.matched_instance_ids} for m in result['matches']],
            'scenes': result['scenes'], 'trainable_parameter_tensors': len(parameters),
            'parameter_groups': groups, 'parameter_details': parameters,
            'frozen_inputs_have_no_grad': all(not v.requires_grad and v.grad is None for v in frozen),
            'labels_have_no_grad': all(not targets[k].requires_grad and targets[k].grad is None for k in ('instance_ids', 'valid_mask', 'confidence')),
            'student_parameter_hash_before': before, 'student_parameter_hash_after': after,
            'reference_state_hash_before': reference_before, 'reference_state_hash_after': reference_after,
            'student_parameters_unchanged': before == after, 'reference_state_unchanged': reference_before == reference_after,
            'optimizer_steps': 0, 'optimizer_created': False, 'real_integration': 'NOT_RUN'}
        EVIDENCE['student_small_backward'] = evidence
        self.assertTrue(evidence['frozen_inputs_have_no_grad'])
        self.assertTrue(evidence['labels_have_no_grad'])
        self.assertEqual(before, after)
        self.assertEqual(reference_before, reference_after)
        for name, group in groups.items():
            self.assertEqual(group['finite_tensors'], group['parameter_tensors'], name)
            self.assertEqual(group['nonzero_tensors'], group['parameter_tensors'], name)

    def test_02_contract_alignment_rejects_mismatch(self):
        cfg = test_student.small_config()
        batch, targets, prediction = synthetic_inputs(cfg.contract_spec)
        validate_training_batch(batch, targets, prediction, spec=cfg.contract_spec)
        for key in ('scene_id', 'frame_ids'):
            wrong = copy.deepcopy(targets)
            wrong[key] = ['other'] if key == 'scene_id' else [['second', 'first']]
            with self.assertRaisesRegex(ContractError, key):
                validate_training_batch(batch, wrong, prediction, spec=cfg.contract_spec)
        batch['metadata']['crop_resize'][0][0]['resize_hw'] = [16, 32]
        with self.assertRaisesRegex(ContractError, 'resize mismatch'):
            validate_training_batch(batch, targets, prediction, spec=cfg.contract_spec)
        EVIDENCE['alignment_limit'] = ('Shared validate_training_batch checks scene/frame/device/shape and validates feature crop metadata. '
            'Targets contain no independent crop provenance; neither this validator nor Prediction alone proves physical crop alignment. '
            'Synthetic labels are generated directly in the same identity-crop grid; real C preprocessing still requires producer evidence.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
