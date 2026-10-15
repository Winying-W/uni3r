"""Strict, purpose-isolated student checkpoint; restore into a new engine only."""
from dataclasses import asdict
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import uuid

import scipy
import torch

from instance_B.b03_utils import ROOT, canonical, config_digest, file_sha256, project_path, write_json
from instance_B.student import InstanceStudent, BASE_SHA256
from instance_B.training import PURPOSE, TrainingEngine, optimizer_group_manifest
from uni3r.instance.contracts import CONTRACT_VERSION

FORMAT_VERSION = 1
MAX_BYTES = 768*1024**2
DPT_SHA256 = 'a47b647c1d545b491dcb649ee509e911eea6d2ebd9b5a43a8642441d955b6b45'
SOURCE_FILES = ('uni3r/instance/contracts.py', 'instance_B/student.py', 'instance_B/dpt_adapter.py',
    'instance_B/matcher.py', 'instance_B/criterion.py', 'instance_B/loss_utils.py',
    'instance_B/training.py', 'instance_B/checkpointing.py', 'instance_B/b03_utils.py')


def provenance():
    return {'contract_version': CONTRACT_VERSION, 'contract_sha256': file_sha256(ROOT/SOURCE_FILES[0]),
            'source_sha256': {rel: file_sha256(ROOT/rel) for rel in SOURCE_FILES},
            'base_checkpoint_sha256': BASE_SHA256, 'dpt_initialization_sha256': DPT_SHA256}


def configs(engine):
    return canonical({'student': engine.model.config.to_dict(), 'loss': asdict(engine.loss_config), 'train': asdict(engine.config)})


def environment():
    return {'python': sys.version, 'torch': str(torch.__version__), 'scipy': str(scipy.__version__),
            'device': 'cpu', 'student_dtype': 'torch.float32', 'loss_dtype': 'torch.float64',
            'intraop': torch.get_num_threads(), 'interop': torch.get_num_interop_threads()}


def boundary_check(engine):
    if not engine.at_boundary or any(p.grad is not None for p in engine.model.parameters()):
        raise ValueError('Checkpoint only at a complete window boundary with no uncommitted gradients')


def build_payload(engine, sample_manifest, generators):
    boundary_check(engine)
    cfg = configs(engine)
    if not generators or any(not isinstance(g, torch.Generator) or g.device.type != 'cpu' for g in generators.values()):
        raise ValueError('Explicit CPU Generator states required')
    return {'format_version': FORMAT_VERSION, 'purpose': PURPOSE, 'training_track': engine.config.training_track,
        'student_state_dict': dict(engine.model.state_dict()), 'optimizer_state_dict': engine.optimizer.state_dict(),
        'optimizer_groups': optimizer_group_manifest(engine.optimizer), 'configs': cfg, 'config_sha256': config_digest(cfg),
        'counters': dict(engine.state), 'window_boundary': True, 'student_training_mode': engine.model.training,
        'rng': {'torch_cpu': torch.get_rng_state(), 'generators': {name: g.get_state() for name, g in generators.items()},
                'python_random_used': False, 'numpy_sampling_used': False},
        'provenance': provenance(), 'samples': canonical(sample_manifest), 'environment': environment()}


def _atomic_publish(path, writer):
    path = project_path(path, write=True)
    if path.exists(): raise FileExistsError(f'Refuse to overwrite checkpoint: {path}')
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    # Exclusive temp and hard-link publication provide no-overwrite atomicity.
    # Only this call's successful temporary link is removed; failures are retained.
    with temp.open('xb') as f:
        writer(f); f.flush(); os.fsync(f.fileno())
    os.link(temp, path)
    temp.unlink()
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def save_checkpoint(engine, path, sample_manifest, generators):
    path = project_path(path, write=True)
    receipt_path = path.with_suffix(path.suffix+'.receipt.json')
    if path.exists() or receipt_path.exists(): raise FileExistsError('Checkpoint/receipt already exists')
    payload = build_payload(engine, sample_manifest, generators)
    _atomic_publish(path, lambda f: torch.save(payload, f))
    if path.stat().st_size > MAX_BYTES: raise ValueError('Unexpectedly large diagnostic checkpoint')
    receipt = {'format_version': FORMAT_VERSION, 'purpose': PURPOSE, 'path': str(path),
        'bytes': path.stat().st_size, 'sha256': file_sha256(path), 'counters': dict(engine.state),
        'config_sha256': payload['config_sha256'], 'provenance': payload['provenance'],
        'optimizer_groups': payload['optimizer_groups'], 'contains_base_model': False,
        'initialization_for_real_or_pseudo': 'FORBIDDEN', 'window_boundary': True}
    encoded = (json.dumps(receipt, indent=2, allow_nan=False)+'\n').encode()
    _atomic_publish(receipt_path, lambda f: f.write(encoded))
    return receipt


def load_verified_payload(path, receipt_path):
    path, receipt_path = project_path(path, write=True), project_path(receipt_path, write=True)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('purpose') != PURPOSE or receipt.get('path') != str(path): raise ValueError('Unknown checkpoint receipt/source/purpose')
    with path.open('rb') as f:
        size = os.fstat(f.fileno()).st_size
        if not 0 < size <= MAX_BYTES or size != receipt.get('bytes'): raise ValueError('Checkpoint size/receipt mismatch')
        sha = hashlib.sha256()
        for block in iter(lambda: f.read(1024*1024), b''): sha.update(block)
        if sha.hexdigest() != receipt.get('sha256'): raise ValueError('Checkpoint SHA256 mismatch')
        f.seek(0)
        payload = torch.load(f, map_location='cpu', weights_only=True)
    return payload, receipt


def _basic_tree(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.layout != torch.strided: raise ValueError('Checkpoint tensors must be dense CPU')
    elif type(value) in (dict,):
        for k, v in value.items():
            if type(k) not in (str, int): raise ValueError('Non-basic checkpoint key')
            _basic_tree(v)
    elif type(value) in (list, tuple):
        for v in value: _basic_tree(v)
    elif value is not None and type(value) not in (str, int, bool, float):
        # torch state_dict is OrderedDict; convert it at save time below.
        raise ValueError(f'Non-basic checkpoint object: {type(value)}')


def _validate_rng(rng, generator_names):
    if not isinstance(rng, dict) or set(rng) != {'torch_cpu', 'generators', 'python_random_used', 'numpy_sampling_used'}:
        raise ValueError('Missing/incompatible random state')
    if rng['python_random_used'] is not False or rng['numpy_sampling_used'] is not False:
        raise ValueError('This diagnostic only uses torch CPU RNG and explicit Generators')
    if set(rng['generators']) != set(generator_names): raise ValueError('Generator name mismatch')
    for state in [rng['torch_cpu'], *rng['generators'].values()]:
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError('Invalid RNG tensor')
        torch.Generator(device='cpu').set_state(state)


def validate_payload(payload, reference_engine, sample_manifest, generators):
    """Pure validation: does not load model, optimizer, counters, or live RNG."""
    required = {'format_version', 'purpose', 'training_track', 'student_state_dict', 'optimizer_state_dict',
        'optimizer_groups', 'configs', 'config_sha256', 'counters', 'window_boundary', 'student_training_mode',
        'rng', 'provenance', 'samples', 'environment'}
    if type(payload) is not dict or set(payload) != required: raise ValueError('Missing/unexpected checkpoint keys')
    _basic_tree(payload)
    if payload['format_version'] != FORMAT_VERSION or payload['purpose'] != PURPOSE or payload['training_track'] != reference_engine.config.training_track:
        raise ValueError('Checkpoint format/purpose/training_track mismatch; real/pseudo initialization forbidden')
    if payload['window_boundary'] is not True: raise ValueError('Mid-window checkpoint unsupported')
    if type(payload['student_training_mode']) is not bool: raise ValueError('Invalid student mode')
    if payload['provenance'] != provenance(): raise ValueError('Contract/base/DPT/source provenance mismatch')
    expected_cfg = configs(reference_engine)
    if payload['configs'] != expected_cfg or payload['config_sha256'] != config_digest(expected_cfg):
        raise ValueError('Complete configuration mismatch (student/loss/train)')
    if payload['samples'] != canonical(sample_manifest): raise ValueError('Synthetic sample/source/cursor protocol mismatch')
    if payload['environment'] != environment(): raise ValueError('Exact resume environment mismatch')
    counters = payload['counters']
    if not isinstance(counters, dict) or set(counters) != set(reference_engine.state): raise ValueError('Missing counters/data cursor')
    if any(type(v) is not int or v < 0 for v in counters.values()): raise ValueError('Invalid counter')
    if counters['skipped_scenes'] > counters['seen_scenes'] or counters['data_cursor'] != counters['seen_scenes'] or counters['global_update_step'] > counters['window_cursor']:
        raise ValueError('Inconsistent counters/data cursor')
    student = payload['student_state_dict']; expected_student = reference_engine.model.state_dict()
    if set(student) != set(expected_student): raise ValueError('Student state keys mismatch; backbone/extra state forbidden')
    for name, expected in expected_student.items():
        value = student[name]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Invalid student tensor shape/dtype/finiteness: {name}')
    expected_groups = optimizer_group_manifest(reference_engine.optimizer)
    if payload['optimizer_groups'] != expected_groups: raise ValueError('Optimizer ordered names/hyperparameters mismatch')
    opt = payload['optimizer_state_dict']
    if set(opt) != {'state', 'param_groups'} or len(opt['param_groups']) != len(expected_groups): raise ValueError('Optimizer group/state structure mismatch')
    ids_to_parameters = {}
    for saved, expected, live in zip(opt['param_groups'], expected_groups, reference_engine.optimizer.param_groups):
        saved_metadata = {k: v for k, v in saved.items() if k != 'params'}
        expected_metadata = {k: v for k, v in expected.items() if k not in ('tensor_count', 'element_count')}
        if canonical(saved_metadata) != expected_metadata: raise ValueError('Optimizer parameter name order/hyperparameters mismatch')
        if len(saved['params']) != len(live['params']): raise ValueError('Optimizer parameter count mismatch')
        for index, param in zip(saved['params'], live['params']):
            if type(index) is not int or index in ids_to_parameters: raise ValueError('Optimizer duplicate/invalid parameter IDs')
            ids_to_parameters[index] = param
    if not set(opt['state']).issubset(ids_to_parameters): raise ValueError('Unknown optimizer parameter state')
    for index, state in opt['state'].items():
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}: raise ValueError('Invalid AdamW state keys')
        step = state['step']
        if not isinstance(step, torch.Tensor) or step.ndim != 0 or step.dtype != torch.float32 or not bool(torch.isfinite(step)):
            raise ValueError('Invalid local AdamW step tensor')
        number = step.item()
        if number != int(number) or not 0 < number <= counters['global_update_step']:
            raise ValueError('Invalid local AdamW step range; not required to equal global update')
        for key in ('exp_avg', 'exp_avg_sq'):
            value, param = state[key], ids_to_parameters[index]
            if not isinstance(value, torch.Tensor) or value.shape != param.shape or value.dtype != param.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError(f'Invalid optimizer {key} shape/dtype/finiteness')
            if key == 'exp_avg_sq' and bool((value < 0).any()): raise ValueError('Negative AdamW second moment')
    _validate_rng(payload['rng'], generators)
    return True


def restore_checkpoint(path, receipt_path, reference_engine, initialization, sample_manifest, generators, *, event_log=None):
    boundary_check(reference_engine)
    payload, receipt = load_verified_payload(path, receipt_path)
    validate_payload(payload, reference_engine, sample_manifest, generators)
    if receipt['config_sha256'] != payload['config_sha256'] or receipt['provenance'] != payload['provenance'] or receipt['counters'] != payload['counters']:
        raise ValueError('Receipt/payload state summary mismatch')
    old_rng = torch.get_rng_state(); old_generators = {name: g.get_state() for name, g in generators.items()}
    try:
        model = InstanceStudent(reference_engine.model.config, dpt_init=initialization)
        candidate = TrainingEngine(model, reference_engine.ledger, loss_config=reference_engine.loss_config,
            train_config=reference_engine.config, event_log=event_log)
        candidate.model.load_state_dict(payload['student_state_dict'], strict=True)
        candidate.optimizer.load_state_dict(payload['optimizer_state_dict'])
        candidate.state = dict(payload['counters']); candidate.model.train(payload['student_training_mode'])
        # Only the fully validated new candidate is returned. No live engine mutation.
        for name, state in payload['rng']['generators'].items(): generators[name].set_state(state)
        torch.set_rng_state(payload['rng']['torch_cpu'])
        return candidate, receipt
    except BaseException:
        torch.set_rng_state(old_rng)
        for name, state in old_generators.items(): generators[name].set_state(state)
        raise
