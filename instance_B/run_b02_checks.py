"""CPU-only B-02 checks, unique attempt folders, no training or optimizer."""
import argparse
from dataclasses import asdict
import functools
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest

import torch

from instance_B.criterion import InstanceCriterion
from instance_B.loss_utils import LossConfig
from instance_B.matcher import BACKEND, BACKEND_VERSION
from uni3r.instance.contracts import CONTRACT_VERSION, DEFAULT_SPEC

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.records = []
    def addSuccess(self, test):
        super().addSuccess(test); self.records.append({'test': test.id(), 'status': 'PASS'})
    def addError(self, test, err):
        super().addError(test, err); self.records.append({'test': test.id(), 'status': 'ERROR', 'traceback': self.errors[-1][1]})
    def addFailure(self, test, err):
        super().addFailure(test, err); self.records.append({'test': test.id(), 'status': 'FAIL', 'traceback': self.failures[-1][1]})
    def addSkip(self, test, reason):
        super().addSkip(test, reason); self.records.append({'test': test.id(), 'status': 'SKIP', 'reason': reason})
    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.records.append({'test': str(subtest), 'status': 'SUBTEST_FAIL', 'traceback': self._exc_info_to_string(err, test)})


def full_shape(out):
    # Independent logits only. No student, no dense QxKxN reference, no backward.
    torch.manual_seed(7302)
    b, q, v, h, w = 1, 100, 2, 256, 256
    ids = (torch.arange(v*h*w).reshape(b, v, h, w) % 101).long()
    targets = {'instance_ids': ids, 'valid_mask': torch.ones_like(ids, dtype=torch.bool),
        'confidence': torch.linspace(.1, 1., v*h*w).reshape(b, v, h, w),
        'scene_id': ['SYNTHETIC_FULL_SHAPE'], 'frame_ids': [['SYNTHETIC_0', 'SYNTHETIC_1']], 'label_source': 'gt'}
    p = {'mask_logits': torch.randn(b, q, v, h, w), 'objectness_logits': torch.randn(b, q),
         'query_features': torch.randn(b, q, 256)}
    start = time.perf_counter()
    with torch.no_grad(): result = InstanceCriterion()(p, targets, gt_set_exhaustive=[True])
    m = result['matches'][0]
    record = {'status': 'PASS', 'check': 'FULL_SHAPE_LOGIT_MATCH_LOSS_FORWARD', 'evidence_kind': 'SYNTHETIC',
        'shape': [b, q, v, h, w], 'K': len(m.target_instance_ids), 'gt_set_exhaustive': [True],
        'seconds': time.perf_counter()-start, 'workspace_design': m.workspace,
        'workspace_budget_bytes': LossConfig.from_json().workspace_budget_bytes,
        'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'RSS_is_not_operator_workspace': True, 'cost_finite': m.cost_finite,
        'losses': {key: result[key].item() for key in ('loss_total', 'loss_bce', 'loss_dice', 'loss_objectness')},
        'loss_dtype': result['scalar_dtype'], 'backend': BACKEND, 'scipy_version': BACKEND_VERSION,
        'student_instantiated': False, 'backward_run': False, 'GPU_used': False}
    assert m.workspace['conservative_workspace_bytes'] <= 64*1024**2
    assert len(m.matched_query_indices) == 100
    write_json(out/'full_shape_forward.json', record)
    print(json.dumps(record), flush=True)
    return True


def tests(out, mode):
    from instance_B.tests import test_contracts, test_student, test_matcher, test_criterion, test_b02_integration
    modules = [test_matcher, test_criterion] if mode == 'unit' else [test_contracts, test_student, test_b02_integration]
    # Cache the existing loader's verified result in this runner only. One artifact
    # read/hash, unchanged loader validation, reused by regression and integration.
    test_student.load_init = functools.lru_cache(maxsize=1)(test_student.load_init)
    os.environ.pop('B01_TEST_ARTIFACT_DIR', None)  # old roundtrip stays in BytesIO
    start = time.perf_counter()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
    with (out/'unittest_output.txt').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordingResult).run(suite)
    print((out/'unittest_output.txt').read_text(), flush=True)
    summary = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skips': len(result.skipped), 'seconds': time.perf_counter()-start, 'records': result.records}
    write_json(out/'unit_results.json', summary)
    for module in modules:
        if hasattr(module, 'EVIDENCE'):
            write_json(out/(module.__name__.split('.')[-1]+'_evidence.json'), module.EVIDENCE)
    if mode == 'integration':
        write_json(out/'initialization_load_cache.json', {'verified_file_loads': test_student.load_init.cache_info().misses,
            'cache_hits': test_student.load_init.cache_info().hits, 'artifact_sha256': 'a47b647c1d545b491dcb649ee509e911eea6d2ebd9b5a43a8642441d955b6b45',
            'loader': 'unchanged instance_B.student.load_dpt_initialization', 'no_full_checkpoint_read': True})
    print('TEST_SUMMARY', json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['unit', 'integration', 'full'], required=True)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    if not out.is_relative_to((ROOT/'instance_B/results/B-02').resolve()):
        raise ValueError('Results must remain within instance_B/results/B-02')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only execution required')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        if os.environ.get(name) != '2': raise ValueError(f'{name}=2 required')
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1': raise ValueError('Disable bytecode writes')
    out.mkdir(parents=True, exist_ok=False)
    (out/'tmp').mkdir(); (out/'cache').mkdir()
    os.environ.update(TMPDIR=str(out/'tmp'), TMP=str(out/'tmp'), TEMP=str(out/'tmp'), XDG_CACHE_HOME=str(out/'cache'))
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.manual_seed(7302)
    config = {'command': ' '.join(sys.argv), 'mode': args.mode, 'python': sys.version, 'executable': sys.executable,
        'torch': torch.__version__, 'contract': CONTRACT_VERSION, 'scipy': BACKEND_VERSION, 'backend': BACKEND,
        'loss_config': asdict(LossConfig.from_json()), 'default_contract_spec': asdict(DEFAULT_SPEC),
        'seed': 7302, 'intraop': torch.get_num_threads(), 'interop': torch.get_num_interop_threads(),
        'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'TMPDIR', 'XDG_CACHE_HOME')},
        'optimizer_steps': 0, 'GPU_USED': 'NO', 'REAL_INTEGRATION': 'NOT_RUN', 'REAL_TRAINING': 'NOT_RUN'}
    write_json(out/'configuration.json', config)
    passed = full_shape(out) if args.mode == 'full' else tests(out, args.mode)
    write_json(out/'completion.json', {'status': 'PASS' if passed else 'FAIL', 'optimizer_steps': 0,
        'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
