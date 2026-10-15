"""Build the teacher-benchmark data cache from SIU3R train archives (validation scans).

CPFS (FUSE) stalls when tens of thousands of small files are extracted concurrently, so we
stream each archive and extract only what the benchmark needs to a local cache:
  pass 1: iou.pt, intrinsic.txt, frame list -> sample pairs (SIU3R training rule, seed 0)
  pass 2: color/depth/extrinsic for every 10th frame (MaskClustering default stride) and
          color/depth/extrinsic/instance/semantic for every context frame.
Output: <cache>/<scan>/{color,depth,extrinsic,instance,semantic}/..., intrinsic.txt, iou.pt
        and pairs.json (written by the parent).
"""
import io
import json
import os
import random
import sys
import tarfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('DATA_ROOT', ROOT / 'data'))
ARCH = str(DATA / 'siu3r_trainval_gt/archives')
CACHE = os.environ.get('TEACHER_BENCH_DATA', str(ROOT / '.cache/teacher_bench_data'))
SCANS = (DATA / 'splits/val_scannet.txt').read_text().split()
PAIRS_PER_SCAN = 5


def sample_pairs(scan, items, iou):
    """Identical to SIU3R ScanNetDataset training sampler, fixed seed per scan."""
    rng = random.Random(f'{scan}-0')
    got, tries = set(), 0
    while len(got) < PAIRS_PER_SCAN and tries < 2000:
        tries += 1
        i1 = rng.randrange(len(items))
        c1 = items[i1]
        cands = [c for c in items[i1 + 10:i1 + 101] if 0.3 < float(iou[c1, c]) < 0.8]
        if cands:
            c2 = rng.choice(cands)
            got.add((c1, c2, round(float(iou[c1, c2]), 4)))
    return [{'scan': scan, 'context_ids': [a, b], 'iou': v} for a, b, v in sorted(got)]


def process(scan):
    out = f'{CACHE}/{scan}'
    done = f'{out}/.done'
    if os.path.exists(done):
        return json.load(open(done))
    os.makedirs(out, exist_ok=True)
    arc = f'{ARCH}/{scan}.tar.gz'
    items = []
    with tarfile.open(arc, 'r|gz') as t:  # pass 1 (streaming)
        for m in t:
            rel = m.name.split('/', 1)[-1]
            if rel in ('iou.pt', 'intrinsic.txt'):
                open(f'{out}/{rel}', 'wb').write(t.extractfile(m).read())
            elif rel.startswith('depth/') and rel.endswith('.png'):
                items.append(int(rel[6:-4]))
    items.sort()
    iou = torch.load(f'{out}/iou.pt', weights_only=True)
    pairs = sample_pairs(scan, items, iou)
    ctx = {c for p in pairs for c in p['context_ids']}
    need = {f for f in items if f % 10 == 0} | ctx
    with tarfile.open(arc, 'r|gz') as t:  # pass 2
        for m in t:
            parts = m.name.split('/')
            if len(parts) != 3 or not m.isfile():
                continue
            sub, fname = parts[1], parts[2]
            fid = int(fname.split('.')[0])
            if (sub in ('color', 'depth', 'extrinsic') and fid in need) or \
               (sub in ('instance', 'semantic') and fid in ctx):
                os.makedirs(f'{out}/{sub}', exist_ok=True)
                open(f'{out}/{sub}/{fname}', 'wb').write(t.extractfile(m).read())
    res = {'pairs': pairs, 'frames': len(need), 'items': len(items)}
    json.dump(res, open(done, 'w'))
    return res


if __name__ == '__main__':
    scans = [s for s in SCANS if os.path.exists(f'{ARCH}/{s}.tar.gz')]
    print(f'{len(scans)}/{len(SCANS)} archives present', flush=True)
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        results = dict(zip(scans, ex.map(process, scans)))
    if len(scans) == len(SCANS):
        pairs = [p for s in SCANS for p in results[s]['pairs']]
        json.dump(pairs, open(f'{CACHE}/pairs.json', 'w'), indent=1)
        print('pairs', len(pairs), 'frames', sum(r['frames'] for r in results.values()))
    else:
        print('partial run; pairs.json not written')
