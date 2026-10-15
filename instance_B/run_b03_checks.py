"""Sequential B-03 runner. Budgeted synthetic steps and isolated real no-grad probe."""
import argparse
import gc
import io
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import unittest

import torch

from instance_B.b03_utils import ROOT, B03_ROOT, project_path, write_json, tensor_fingerprints, tree_digest, config_digest
from instance_B.training import BudgetLedger, TrainingEngine, optimizer_group_manifest
from instance_B.student import InstanceStudent, load_dpt_initialization
from instance_B.checkpointing import (save_checkpoint, build_payload, load_verified_payload, validate_payload,
    restore_checkpoint, configs, environment, provenance)
from instance_B.run_b02_checks import RecordingResult
from instance_B.tests import test_contracts, test_student, test_matcher, test_criterion, test_b02_integration, test_training, test_checkpointing
from instance_B.probe_m0 import probe

SCHEDULE = [('W1', ['complete', 'partial', 'background', 'ignore'], 0),
            ('I', ['ignore', 'background_partial'], 4),
            ('W2', ['complete', 'partial'], 6), ('W3', ['complete'], 8)]


def progress(message):
    path = ROOT/'instance_B/B_PROGRESS.md'
    if not path.resolve().is_relative_to(ROOT/'instance_B'): raise ValueError('Progress boundary')
    before = path.read_bytes()
    if path.read_bytes() != before: raise RuntimeError('Concurrent progress change')
    with path.open('a', encoding='utf-8') as f: f.write('\n- B03_IN_PROGRESS: '+message+'\n')


def suite(modules, out, label):
    started = time.perf_counter()
    tests = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
    with (out/(label+'_stdout.txt')).open('x') as f:
        result = unittest.TextTestRunner(stream=f, verbosity=2, resultclass=RecordingResult).run(tests)
    summary = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skips': len(result.skipped), 'seconds': time.perf_counter()-started, 'records': result.records,
        'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out/(label+'_results.json'), summary)
    for module in modules:
        if hasattr(module, 'EVIDENCE'): write_json(out/(label+'_'+module.__name__.split('.')[-1]+'_evidence.json'), module.EVIDENCE)
    print(label, json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)
    if not result.wasSuccessful(): print((out/(label+'_stdout.txt')).read_text(), flush=True)
    return result.wasSuccessful()


def evaluate(model, batch):
    mode = model.training; model.eval()
    try:
        with torch.no_grad(): return {k: v.clone() for k, v in model(batch).items()}
    finally: model.train(mode)


def engine_snapshot(engine, generators):
    return {'student': tensor_fingerprints(dict(engine.model.state_dict())),
        'optimizer_tensors': tensor_fingerprints(engine.optimizer.state_dict()),
        'optimizer_digest': tree_digest(engine.optimizer.state_dict()),
        'state': dict(engine.state), 'groups': optimizer_group_manifest(engine.optimizer),
        'torch_rng': tensor_fingerprints(torch.get_rng_state()),
        'generator_rng': {k: tensor_fingerprints(g.get_state()) for k, g in generators.items()}}


def next_window(engine, generators, index):
    label, kinds, cursor = SCHEDULE[index]
    if engine.state['data_cursor'] != cursor or engine.state['window_cursor'] != index:
        raise ValueError('Data/window cursor does not match the fixed diagnostic schedule')
    return [test_training.make_example(kind, generators['data'], f'{label}_{i}') for i, kind in enumerate(kinds)]


def run_diagnostic(initialization, ledger, out):
    if ledger.physical_calls+5 > 8: raise RuntimeError('Insufficient remaining stage budget for five-call resume protocol')
    torch.manual_seed(7303)
    generators = {'data': torch.Generator().manual_seed(73031), 'eval': torch.Generator().manual_seed(73039)}
    samples = {'purpose': 'SYNTHETIC_ENGINE_DIAGNOSTIC_NOT_FOR_REAL_TRAINING', 'data_generator_seed': 73031,
        'eval_generator_seed': 73039, 'torch_seed': 7303, 'source': 'B-private synthetic tensors in shared contract; no real cache',
        'ordered_windows': [{'name': name, 'kinds': kinds, 'cursor_start': cursor,
            'gt_set_exhaustive': [kind in ('complete', 'background', 'ignore', 'zero_support') for kind in kinds]}
            for name, kinds, cursor in SCHEDULE], 'geometry_confidence': 'synthetic nonnegative model score; not loss weight'}
    write_json(out/'synthetic_sample_manifest.json', samples)
    evaluation = test_training.make_example('complete', generators['eval'], 'FIXED_EVAL')[0]
    model = InstanceStudent(test_student.small_config(), dpt_init=initialization)
    engine = TrainingEngine(model, ledger, event_log=out/'continuous_events.jsonl')
    write_json(out/'effective_configuration.json', {'configs': configs(engine), 'environment': environment(),
        'provenance': provenance(), 'groups': optimizer_group_manifest(engine.optimizer), 'samples': samples})
    # Full safe serialization/validation at zero optimizer steps, before spending budget.
    payload = build_payload(engine, samples, generators)
    buffer = io.BytesIO(); torch.save(payload, buffer); buffer.seek(0)
    loaded = torch.load(buffer, map_location='cpu', weights_only=True)
    validate_payload(loaded, engine, samples, generators)
    del payload, loaded, buffer; gc.collect()
    init_before = tensor_fingerprints(dict(initialization.state))
    first = next_window(engine, generators, 0)
    input_before = tree_digest(first)
    original_parameters = tensor_fingerprints(dict(model.state_dict()))
    w1 = engine.run_window(first, label='CONTINUOUS_W1')
    assert tree_digest(first) == input_before
    assert all(not t.requires_grad and t.grad is None for sample in first for t in [sample[0]['images'], sample[0]['means'], sample[0]['geometry_confidence'],
        *sample[0]['patch_tokens'].values(), sample[1]['confidence']])
    changed = tensor_fingerprints(dict(model.state_dict()))
    changed_names = [name for name in changed if changed[name] != original_parameters[name]]
    assert any(n.startswith('.instance_dpt.') for n in changed_names)
    assert any(not n.startswith('.instance_dpt.') for n in changed_names)
    assert all(g['finite'] == g['with_grad'] and g['nonzero_tensors'] > 0 for g in w1['gradient_groups'].values())
    assert all(abs(v['delta_minus_weight_decay_only']) > 1e-10 for v in w1['update_anchors'].values())
    w1['changed_parameter_names'] = changed_names
    w1['inputs_labels_unchanged_and_no_grad'] = True
    w1['DPT_initialization_unchanged'] = tensor_fingerprints(dict(initialization.state)) == init_before
    write_json(out/'student_optimizer_update.json', w1)
    saved_outputs = evaluate(model, evaluation)
    path = out/'synthetic_engine_W1.pt'
    receipt = save_checkpoint(engine, path, samples, generators)
    saved_state = engine_snapshot(engine, generators)
    write_json(out/'checkpoint_state_summary.json', saved_state)
    payload, _ = load_verified_payload(path, path.with_suffix('.pt.receipt.json'))
    test_checkpointing.CONTEXT.update(payload=payload, engine=engine, samples=samples, generators=generators,
        out=out, path=path, receipt=receipt, initialization=initialization)
    check_passed = suite([test_checkpointing], out, 'checkpoint_tests')
    test_checkpointing.CONTEXT.clear(); del payload; gc.collect()
    if not check_passed: raise RuntimeError('Checkpoint negative tests failed; no further optimizer steps')
    progress('Checkpoint strict payload/safe loading and rejection tests completed; W1 checkpoint saved; physical calls='+str(ledger.physical_calls)+'. Resume equivalence next. Evidence: '+str(out))
    rng_witness = torch.rand(4)
    ignored = next_window(engine, generators, 1)
    skip_before = engine_snapshot(engine, generators)
    calls_before_skip = ledger.physical_calls
    skipped = engine.run_window(ignored, label='CONTINUOUS_I')
    skip_after = engine_snapshot(engine, generators)
    for key in ('student', 'optimizer_tensors', 'optimizer_digest', 'groups', 'torch_rng', 'generator_rng'):
        assert skip_before[key] == skip_after[key], key
    assert skip_before['state']['global_update_step'] == skip_after['state']['global_update_step'] == 1
    assert calls_before_skip == ledger.physical_calls and skipped['forward_count'] == 0
    assert engine.optimizer.state, 'Skip must be tested after AdamW moments exist'
    write_json(out/'no_supervision_skip_after_adam_state.json', {'status': 'PASS', 'before_counters': skip_before['state'],
        'after_counters': skip_after['state'], 'physical_calls_unchanged': True,
        'parameters_buffers_moments_steps_LR_unchanged': True, 'report': skipped})
    continuous_windows = []
    for index in (2, 3):
        items = next_window(engine, generators, index)
        fingerprint = tree_digest(items)
        result = engine.run_window(items, label='CONTINUOUS_'+SCHEDULE[index][0])
        assert tree_digest(items) == fingerprint
        continuous_windows.append({'input_fingerprint': fingerprint, 'report': result})
        del items
    final_outputs = evaluate(model, evaluation)
    continuous_final = engine_snapshot(engine, generators)
    write_json(out/'continuous_final_fingerprints.json', continuous_final)
    candidate, _ = restore_checkpoint(path, path.with_suffix('.pt.receipt.json'), engine, initialization, samples, generators,
                                     event_log=out/'resumed_events.jsonl')
    assert engine_snapshot(candidate, generators) == saved_state
    restored_outputs = evaluate(candidate.model, evaluation)
    for key in saved_outputs: torch.testing.assert_close(saved_outputs[key], restored_outputs[key], atol=0, rtol=0)
    assert torch.equal(torch.rand(4), rng_witness)
    resumed_ignored = next_window(candidate, generators, 1)
    assert tree_digest(resumed_ignored) == tree_digest(ignored)
    candidate.run_window(resumed_ignored, label='RESUMED_I')
    resumed_windows = []
    for index, expected in zip((2, 3), continuous_windows):
        items = next_window(candidate, generators, index)
        assert tree_digest(items) == expected['input_fingerprint']
        result = candidate.run_window(items, label='RESUMED_'+SCHEDULE[index][0])
        assert result['losses'] == expected['report']['losses']
        resumed_windows.append({'input_fingerprint': tree_digest(items), 'report': result})
        del items
    resumed_final = engine_snapshot(candidate, generators)
    write_json(out/'resumed_final_fingerprints.json', resumed_final)
    if resumed_final != continuous_final:
        write_json(out/'resume_mismatch.json', {'continuous': continuous_final, 'resumed': resumed_final})
        raise AssertionError('Exact resume state mismatch; diagnostic evidence retained')
    candidate_outputs = evaluate(candidate.model, evaluation)
    output_errors = {}
    for key in final_outputs:
        torch.testing.assert_close(final_outputs[key], candidate_outputs[key], atol=0, rtol=0)
        output_errors[key] = (final_outputs[key]-candidate_outputs[key]).abs().max().item()
    assert tensor_fingerprints(dict(initialization.state)) == init_before
    result = {'status': 'PASS', 'checkpoint': str(path), 'checkpoint_purpose': receipt['purpose'],
        'checkpoint_size_bytes': receipt['bytes'], 'checkpoint_sha256': receipt['sha256'],
        'continuous_global_update_step': engine.state['global_update_step'],
        'resumed_global_update_step': candidate.state['global_update_step'],
        'physical_optimizer_step_calls': ledger.physical_calls, 'checkpoint_time_outputs_exact': True,
        'final_student_optimizer_counters_cursor_LR_RNG_exact': True, 'final_raw_output_max_errors': output_errors,
        'continuous_windows': continuous_windows, 'resumed_windows': resumed_windows,
        'DPT_initialization_unchanged': True, 'no_full_backbone_loaded_or_verified': True,
        'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out/'resume_equivalence.json', result)
    del candidate, engine, model, first, ignored, resumed_ignored, evaluation
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    run_root = project_path(args.run_root, write=True)
    out = project_path(run_root/args.attempt, write=True)
    if out.parent != run_root or not args.attempt.startswith('attempt_'): raise ValueError('Expected unique attempt name')
    out.mkdir(exist_ok=False)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        if os.environ.get(name) != '2': raise ValueError(f'{name}=2 required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PYTHONDONTWRITEBYTECODE') != '1': raise ValueError('CPU/no-bytecode environment required')
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7303)
    ledger = BudgetLedger(run_root/'optimizer_budget.jsonl')
    write_json(out/'invocation.json', {'command': ' '.join(sys.argv), 'environment': environment(),
        'budget_ledger': str(ledger.path), 'physical_calls_before': ledger.physical_calls,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'env': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TMPDIR', 'TEMP', 'TMP', 'XDG_CACHE_HOME', 'PYTHONDONTWRITEBYTECODE')}})
    init_dir = project_path(ROOT/'instance_B/results/B-01/20261008_cpu_01')
    initialization = load_dpt_initialization(init_dir/'dpt_init_from_2views.pt', init_dir/'dpt_init_receipt.json')
    init_fingerprint = tensor_fingerprints(dict(initialization.state))
    # Reuse the single existing safe-loader result; old test source stays read-only.
    test_student.load_init = lambda: initialization
    test_training.CONTEXT.update(initialization=initialization, ledger=ledger, out=out)
    outcomes = {}
    outcomes['regression'] = suite([test_contracts, test_student, test_matcher, test_criterion, test_b02_integration], out, 'regression')
    outcomes['training_tests'] = suite([test_training], out, 'training_tests')
    progress('Training engine checks executed: regression='+str(outcomes['regression'])+', training_tests='+str(outcomes['training_tests'])+'. Actual results '+str(out)+'. No optimizer diagnostic starts unless both suites pass.')
    if outcomes['regression'] and outcomes['training_tests']:
        try: outcomes['diagnostic'] = run_diagnostic(initialization, ledger, out)
        except BaseException as exc:
            outcomes['diagnostic'] = {'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc()}
            write_json(out/'diagnostic_failure.json', outcomes['diagnostic'])
            print(outcomes['diagnostic']['traceback'], flush=True)
            test_checkpointing.CONTEXT.clear(); gc.collect()
    else: outcomes['diagnostic'] = {'status': 'NOT_RUN', 'reason': 'Zero-step prerequisite suite failure'}
    try: outcomes['probe'] = probe(initialization, out/'m0_probe')
    except BaseException as exc:
        outcomes['probe'] = {'status': 'BLOCKED', 'error': repr(exc), 'traceback': traceback.format_exc()}
        write_json(out/'probe_failure.json', outcomes['probe'])
        print(outcomes['probe']['traceback'], flush=True)
    progress('A/C probe executed separately from synthetic updates; result='+str(outcomes['probe'].get('status'))+'. C GT admission='+str(outcomes['probe'].get('C_TARGET_ADMISSION', 'see probe failure'))+'. Real training remains NOT_RUN. Evidence '+str(out))
    outcomes['physical_optimizer_step_calls'] = ledger.physical_calls
    outcomes['initialization_file_verified_load_count'] = 1
    outcomes['initialization_tensors_unchanged'] = tensor_fingerprints(dict(initialization.state)) == init_fingerprint
    outcomes['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out/'runner_outcome.json', outcomes)
    print('B03_OUTCOME', json.dumps({k: v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('status', 'error', 'C_TARGET_ADMISSION', 'REAL_GT_LOSS_FORWARD', 'continuous_global_update_step', 'resumed_global_update_step')} for k, v in outcomes.items()}), flush=True)
    return 0 if outcomes['regression'] and outcomes['training_tests'] and outcomes['diagnostic']['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
