"""Compare completed full evaluations with fixed-view paper references."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('metrics', nargs='+', type=Path)
args = parser.parse_args()
refs = {2: {'psnr':25.53, 'ssim':.8727, 'lpips':.1380, 'miou':.5584, 'accuracy':.8268},
        8: {'psnr':24.71, 'ssim':.851, 'miou':.554, 'accuracy':.807},
        16: {'psnr':23.32, 'ssim':.816, 'miou':.558, 'accuracy':.835}}
for path in args.metrics:
    data = json.loads(path.read_text())
    if (data['smoke_run'] or data['num_batches'] != data['total_batches']
            or data.get('shard_count', 1) > 1):
        raise SystemExit(f'Refusing incomplete/smoke results: {path}')
    views = data['input_views']
    print(f"\n{views} views: {data['num_batches']} samples, {data['scene_count']} scenes")
    print(f'Source: {path}')
    if views in (8, 16) and data['scene_count'] != 40:
        print('Protocol caveat: released loader uses 49 scenes; paper section 4.1 states 40.')
    print()
    print('| Metric | Measured | Paper | Difference |')
    print('|---|---:|---:|---:|')
    for key, ref in refs[views].items():
        value = data[key]
        print(f'| {key} | {value:.6f} | {ref:.6f} | {value-ref:+.6f} |')
    print()
    print('Depth is target-view and is not compared with source-view paper depth.')
