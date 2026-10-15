"""B-private paths and compact integrity records; no data decoding or contracts."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
B03_ROOT = ROOT/'instance_B/results/B-03'


def project_path(path, *, write=False):
    p = Path(path).resolve()
    boundary = B03_ROOT.resolve() if write else ROOT
    if not p.is_relative_to(boundary):
        raise ValueError(f'Path outside authorized boundary: {path}')
    return p


def file_sha256(path):
    p = project_path(path)
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def config_digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_fingerprints(value):
    result = {}
    def visit(x, path):
        if isinstance(x, torch.Tensor):
            result[path] = {'shape': list(x.shape), 'dtype': str(x.dtype),
                'sha256': hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        elif isinstance(x, dict):
            for key in sorted(x, key=str): visit(x[key], f'{path}.{key}')
        elif isinstance(x, (list, tuple)):
            for i, item in enumerate(x): visit(item, f'{path}[{i}]')
    visit(value, '')
    return result


def tree_digest(value):
    """Include tensor bytes and all basic scalar/config/counter state."""
    def normalize(x):
        if isinstance(x, torch.Tensor): return tensor_fingerprints(x)['']
        if isinstance(x, dict): return {str(k): normalize(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)): return [normalize(v) for v in x]
        if x is None or type(x) in (str, int, bool, float): return x
        raise TypeError(f'Non-basic snapshot object: {type(x)}')
    return config_digest(normalize(value))


def write_json(path, value):
    path = project_path(path, write=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
