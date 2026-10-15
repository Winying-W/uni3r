"""Audit converted dataset inventories and representative camera records."""
import concurrent.futures
import json
from pathlib import Path
import os
import numpy as np

DATA = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[1] / 'data'))


def check(task):
    kind, marker = task
    report = json.loads(marker.read_text())
    root = marker.parent
    if kind == 'scannet':
        groups = [('color', '*.png'), ('depth', '*.png'), ('pose', '*.npz')]
        camera_key, pose_key = 'camera_intrinsics', 'camera_pose'
    else:
        groups = [('rgb_resized_undistorted', '*.JPG'), ('render_depth', '*.png'), ('camera', '*.npz')]
        camera_key, pose_key = 'intrinsic', 'extrinsic'
    files = [sorted((root / folder).glob(pattern)) for folder, pattern in groups]
    stems = [{p.stem for p in group} for group in files]
    if not (stems[0] == stems[1] == stems[2]) or len(stems[0]) != report['frames']:
        raise RuntimeError(f'Inconsistent frame inventory: {root}')
    if files[2]:
        for idx in {0, len(files[2]) // 2, len(files[2]) - 1}:
            with np.load(files[2][idx], allow_pickle=False) as camera:
                k, pose = camera[camera_key], camera[pose_key]
                if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
                    raise RuntimeError(f'Invalid camera record: {files[2][idx]}')
                if k[0, 0] <= 0 or k[1, 1] <= 0:
                    raise RuntimeError(f'Invalid focal length: {files[2][idx]}')
                if not np.allclose(pose[3], [0, 0, 0, 1]):
                    raise RuntimeError(f'Invalid homogeneous pose: {files[2][idx]}')
    return {'dataset': kind, 'scene': report['scene'], 'frames': report['frames']}


def main():
    scannet = json.loads((DATA / 'scannet_conversion.json').read_text())
    if scannet['max_scenes']:
        raise RuntimeError('ScanNet conversion report is only a smoke test')
    excluded = [s for s in scannet['scenes'] if s.get('status', 'converted') != 'converted']
    if excluded:
        raise RuntimeError(f'Unresolved ScanNet conversion exclusions: {excluded}')
    pp = json.loads((DATA / 'scannetpp_conversion.json').read_text())
    tasks = [('scannet', p) for p in (DATA / 'scannet_processed').glob('*/conversion.json')]
    tasks += [('scannetpp', p) for p in (DATA / 'scannetpp_processed').glob('*/dslr/conversion.json')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(check, tasks))
    expected = {'scannet': {r['scene'] for r in scannet['scenes']},
                'scannetpp': {r['scene'] for r in pp if r['status'] == 'converted'}}
    summary = {'verified': True, 'datasets': {}}
    for kind in expected:
        selected = [r for r in results if r['dataset'] == kind]
        if {r['scene'] for r in selected} != expected[kind]:
            raise RuntimeError(f'Scene inventory mismatch: {kind}')
        summary['datasets'][kind] = {'scenes': len(selected), 'frames': sum(r['frames'] for r in selected),
                                    'loader_eligible_scenes': sum(r['frames'] > (32 if kind == 'scannet' else 128) for r in selected)}
    summary['scannetpp_excluded_short_scenes'] = [r['scene'] for r in pp if r['status'] != 'converted']
    (DATA / 'verification.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
