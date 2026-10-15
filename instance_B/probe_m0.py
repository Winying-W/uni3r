"""One authorized CPU real-cache forward. No A/C executable imports or training."""
import gc
import json
from pathlib import Path
import resource
import time

import torch

from instance_B.b03_utils import ROOT, project_path, file_sha256, tensor_fingerprints, tree_digest, write_json
from instance_B.student import InstanceStudent, StudentConfig, BASE_SHA256, flatten_membership_logits
from uni3r.instance.contracts import validate_frozen_scene_batch, TOKEN_LAYERS


def inspect_c_handoff():
    """Only known progress/index paths; do not infer that another teammate is idle."""
    checked, found = [], []
    for relative in ('instance_C/C_PROGRESS.md', 'instance_C/README.md', 'instance_C/handoff.json',
                     'uni3r/instance/C_PROGRESS.md', 'reproduce/C_PROGRESS.md'):
        path = project_path(ROOT/relative)
        checked.append({'path': relative, 'exists': path.exists()})
        if path.is_file():
            if path.stat().st_size > 1024*1024: raise ValueError('Unexpectedly large C index')
            found.append({'path': relative, 'sha256': file_sha256(path), 'text': path.read_text()})
    return {'status': 'NEEDS_REVIEW' if found else 'BLOCKED', 'checked_paths': checked, 'found_handoffs': found,
            'reason': 'New C handoff requires explicit reading and admission' if found else 'This bounded current check found no C handoff/contract-format GT targets',
            'missing': ['contract-format targets', 'A-matched scene/frame/grid', 'physical cross-view ID statement',
                        'valid/confidence/ignore definition', 'ordered crop/resize evidence', 'source/split/diagnostic purpose',
                        'exhaustiveness basis (otherwise False)']}


def probe(initialization, output):
    output = project_path(output, write=True)
    output.mkdir(exist_ok=False)
    c = inspect_c_handoff()
    write_json(output/'C_admission.json', c)
    directory = project_path(ROOT/'instance_A/results/M0_20261008_01')
    report_path = project_path(directory/'report.json')
    metadata_path = project_path(directory/'metadata.json')
    source_path = project_path(directory/'source_manifest.json')
    fixture_path = project_path(directory/'fixture.pt')
    report = json.loads(report_path.read_text()); metadata = json.loads(metadata_path.read_text())
    if report.get('status') != 'A_M0_REAL_FIXTURE_VERIFIED' or report.get('purpose') != 'M0_DIAGNOSTIC_ONLY_NOT_FORMAL_TRAIN_OR_VAL':
        raise ValueError('A producer receipt/status/purpose not admitted')
    if not 0 < fixture_path.stat().st_size <= 256*1024**2: raise ValueError('A fixture size outside admitted range')
    hashes = {}
    for p in (fixture_path, metadata_path, source_path):
        hashes[p.name] = file_sha256(p)
        if hashes[p.name] != report['artifact_sha256'][p.name]: raise ValueError(f'A artifact receipt mismatch: {p.name}')
    batch = torch.load(fixture_path, map_location='cpu', weights_only=True)
    if not isinstance(batch, dict) or batch.get('metadata') != metadata: raise ValueError('Unknown A wrapper or metadata mismatch; no conversion allowed')
    validate_frozen_scene_batch(batch)
    if metadata['evidence_kind'] != 'real' or metadata['base_checkpoint']['sha256'] != BASE_SHA256:
        raise ValueError('A real official 2views provenance mismatch')
    if metadata['purpose'] != report['purpose'] or metadata['scene_id'] != report['scene_id'] or metadata['frame_ids'] != report['frame_ids']:
        raise ValueError('A receipt metadata order/purpose mismatch')
    if metadata['uni3r_source_sha256'] != hashes['source_manifest.json'] or metadata['base_checkpoint']['name'] != '2views.pth':
        raise ValueError('A source fingerprint/base name mismatch')
    if batch['images'].shape[0] != 1 or metadata['rgb_normalization'] != 'minus_one_to_one' or list(metadata['token_layers']) != list(TOKEN_LAYERS):
        raise ValueError('A profile/RGB/token ordering mismatch')
    if not report.get('all_base_parameters_frozen') or not report.get('base_train_protected') or report['gradient_isolation'].get('base_gradients') != 0:
        raise ValueError('Required producer freezing evidence missing')
    if not report['gradient_isolation'].get('base_parameter_versions_unchanged'):
        raise ValueError('Producer base version isolation evidence missing')
    for scene in metadata['crop_resize']:
        for crop in scene:
            if not crop.get('operations') or not crop.get('source_rgb_sha256') or not crop.get('source_camera_npz_sha256'):
                raise ValueError('Incomplete producer crop/source evidence')
    if not metadata['gaussian_schema'].startswith('A.raw_views.v1:'):
        raise ValueError('Unknown Gaussian schema')
    before = tensor_fingerprints(batch); before_all = tree_digest(batch)
    for name, item in before.items():
        if item['dtype'].startswith('torch.float') and item['dtype'] != 'torch.float32':
            raise ValueError(f'A fp32 profile violated by {name}; no casting')
    v, h, w = 2, 256, 256
    indices = batch['gaussians']['source_indices']
    if len(indices) != 1 or not torch.equal(indices[0], torch.arange(v*h*w)):
        raise ValueError('A unfiltered source_indices does not match declared mapping')
    for view in range(v):
        if not torch.equal(batch['gaussians']['views'][view]['means'], batch['means'][:, view]):
            raise ValueError('A Gaussian/source means correspondence failed')
        for key, tail in {'scales': (3,), 'rotations': (4,), 'covs': (3, 3), 'opacities': (1,),
                          'sh_coeffs': (16, 3), 'gs_feats': (64,)}.items():
            if tuple(batch['gaussians']['views'][view][key].shape) != (1, h, w, *tail):
                raise ValueError(f'A raw Gaussian schema shape mismatch: {key}')
    admission = {'status': 'PASS', 'tensor_validation': 'PASS', 'producer_evidence': 'PRESENT_AND_CROSS_CHECKED',
        'producer_evidence_not_independently_rerun': True, 'fixture_path': str(fixture_path), 'fixture_bytes': fixture_path.stat().st_size,
        'artifact_hashes': hashes, 'metadata': metadata, 'input_tensors': before,
        'original_source_paths_in_metadata_not_followed': True, 'data_use': report['purpose'],
        'no_backbone_load': True, 'no_A_C_code_execution': True}
    write_json(output/'A_admission.json', admission)
    torch.manual_seed(7303)
    model = InstanceStudent(StudentConfig(), dpt_init=initialization).eval()
    model_before = tensor_fingerprints(dict(model.state_dict()))
    start = time.perf_counter()
    with torch.no_grad():
        prediction = model(batch)
        flattened = flatten_membership_logits(prediction['mask_logits'], indices)[0]
        # All source indices, not a random subset. View-major/y/x order is explicit.
        direct = torch.stack([prediction['mask_logits'][0, :, view].reshape(100, h*w) for view in range(v)], dim=1).reshape(100, v*h*w)
        if not torch.equal(flattened, direct): raise AssertionError('Flatten/source_indices correspondence failed')
        outputs = {k: {'shape': list(x.shape), 'dtype': str(x.dtype), 'finite': bool(torch.isfinite(x).all()),
                       'min': x.min().item(), 'max': x.max().item(), 'mean': x.double().mean().item(),
                       'sha256': tensor_fingerprints(x)['']['sha256']} for k, x in prediction.items()}
    unchanged = tensor_fingerprints(batch) == before and tree_digest(batch) == before_all
    unchanged_model = tensor_fingerprints(dict(model.state_dict())) == model_before
    if not unchanged or not unchanged_model: raise AssertionError('Probe changed cached input/model state')
    outcome = {'status': 'PASS', 'REAL_STUDENT_FORWARD': 'PASS', 'A_FIXTURE_ADMISSION': 'PASS',
        'seconds': time.perf_counter()-start, 'outputs': outputs, 'student_config': model.config.to_dict(),
        'fresh_student_no_synthetic_updates': True, 'student_mode': 'eval', 'grad_mode': 'no_grad',
        'input_and_Gaussian_payload_unchanged': unchanged, 'student_parameters_and_buffers_unchanged': unchanged_model,
        'flatten_source_indices_all_pixels': 'PASS', 'real_data_backward_calls': 0, 'real_data_optimizer_steps': 0,
        'C_TARGET_ADMISSION': c['status'], 'REAL_GT_LOSS_FORWARD': 'BLOCKED',
        'real_gt_reason': c['reason'], 'REAL_TRAINING': 'NOT_RUN',
        'online_Uni3R_path_not_rerun': True, 'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(output/'real_student_forward.json', outcome)
    del model, batch, prediction, flattened, direct
    gc.collect()
    return outcome
