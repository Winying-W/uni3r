"""Merge disjoint evaluation shards using metric sufficient statistics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uni3r.utils.path_manager import init_all_submodules
init_all_submodules()
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.segmentation import MeanIoU
from torchmetrics import Accuracy
from uni3r.loss import DepthEstimationMetric


def merge(shards, expected_samples, prefix='TestDataset'):
    reports = [json.loads((s / 'metrics.json').read_text()) for s in shards]
    for field in ('checkpoint', 'dataset', 'criterion', 'shard_count'):
        if len({r[field] for r in reports}) != 1:
            raise ValueError(f'Shard settings disagree: {field}')
    if any(r['max_batches'] for r in reports):
        raise ValueError('Refusing to report a smoke test as full evaluation')
    if sorted(r['shard_index'] for r in reports) != list(range(reports[0]['shard_count'])):
        raise ValueError('Missing or duplicate shard')
    blobs = [torch.load(str(s / f'{prefix}_metric_states.pt'), map_location='cpu') for s in shards]
    indices = [idx for b in blobs for idx in b['sample_indices']]
    if sorted(indices) != list(range(expected_samples)):
        raise ValueError('Samples are missing, duplicated, or out of range')
    metrics = {'psnr': PeakSignalNoiseRatio(data_range=1.),
               'ssim': StructuralSimilarityIndexMeasure(data_range=1.),
               'miou': MeanIoU(num_classes=9, include_background=False, per_class=True, input_format='index'),
               'accuracy': Accuracy(num_classes=9, task='multiclass', ignore_index=0),
               'depth_metric': DepthEstimationMetric(align_by_median=False, align_by_least_squares=True)}
    values = {}
    for name, metric in metrics.items():
        for key in metric._defaults:
            entries = torch.stack([b['states'][name][key] for b in blobs])
            # data_range is a constant; every other state is additive.
            value = entries.mean(0) if name == 'psnr' and key == 'data_range' else entries.sum(0)
            setattr(metric, key, value)
        metric._update_count = sum(b['updates'][name] for b in blobs)
        value = metric.compute()
        values[name] = ({k: v.tolist() for k, v in value.items()} if isinstance(value, dict) else value.tolist())
    values['iou_per_class'] = values['miou']
    values['miou'] = sum(values['iou_per_class']) / len(values['iou_per_class'])
    lpips = [v for b in blobs for v in b['lpips_scores']]
    times = [v for b in blobs for v in b['inference_times']]
    values['lpips'] = sum(lpips) / len(lpips)
    values['average_inference_seconds'] = sum(times) / len(times) if times else None
    return {**{k: reports[0][k] for k in ('checkpoint', 'dataset', 'criterion')},
            'samples': expected_samples, 'shards': len(shards), 'metrics': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('shards', nargs='+', type=Path)
    parser.add_argument('--expected-samples', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', default='TestDataset')
    args = parser.parse_args()
    result = merge(args.shards, args.expected_samples, args.prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
