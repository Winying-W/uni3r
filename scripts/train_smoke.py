"""Run the upstream training loop on one real batch from each dataset."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as entry  # installs Uni3R's losses in the upstream training loop
import torch
import yaml
from dust3r.training import train_one_epoch
from croco.utils.misc import NativeScalerWithGradNormCount, get_parameter_groups
from uni3r.uni3r import Uni3R
from uni3r.loss import GaussianLoss
from uni3r.datasets.scannet import Scannet
from uni3r.datasets.scannetpp import Scannetpp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('test_results/train_smoke.json'))
    args = p.parse_args()
    torch.manual_seed(777)
    root = Path('data').resolve()
    selected = [('scannet', root / 'scannet_processed/scene0001_01', Scannet),
                ('scannetpp', root / 'scannetpp_processed/036bce3393', Scannetpp)]
    datasets = []
    for name, scene, cls in selected:
        if not scene.exists():
            raise FileNotFoundError(scene)
        local = root / 'training_smoke' / name
        local.mkdir(parents=True, exist_ok=True)
        link = local / scene.name
        if not link.exists():
            link.symlink_to(scene, target_is_directory=True)
        d = cls(ROOT=str(local), split='train', num_views=2, resolution=(256, 256), seed=777)
        datasets.append(torch.utils.data.Subset(d, [0]))
    loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=1, num_workers=0)
    model = Uni3R(yaml.safe_load(Path('configs/default.yaml').read_text())).cuda()
    criterion = GaussianLoss().cuda()
    optimizer = torch.optim.AdamW(get_parameter_groups(model, .05), lr=1e-4, betas=(.9, .95))
    config = argparse.Namespace(accum_iter=1, amp=0, print_freq=1, lr=1e-4,
                                min_lr=1e-6, warmup_epochs=0, epochs=1)
    before = model.gs_attr_proj[-1].weight.detach().clone()
    stats = train_one_epoch(model, criterion, loader, optimizer, torch.device('cuda'), 0,
                           NativeScalerWithGradNormCount(enabled=False), config)
    weight = model.gs_attr_proj[-1].weight.detach()
    change = (weight - before).abs().max().item()
    if not torch.isfinite(weight).all() or change == 0:
        raise RuntimeError('Training did not produce a finite parameter update')
    report = {'batches': 2, 'datasets': ['ScanNet', 'ScanNet++'], 'optimizer': 'AdamW',
              'amp': False, 'max_parameter_change': change, 'stats': stats,
              'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(),
              'scope': 'Training smoke test, not full 100-epoch training'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
