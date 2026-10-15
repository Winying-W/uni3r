"""SIU3R ScanNet val data fed through the exact Uni3R 2-view TestDataset protocol.

Used for a distribution check of the official Uni3R weights: same crop/resize code,
same camera normalisation, same view order (source1, source2, target), same metrics.

Each SIU3R val pair gives two context frames and several target frames; every target
that is not a context frame becomes one sample, mirroring LSM's one-target-per-sample.

Semantic labels: SIU3R uses the 20 ScanNet classes (0 = unlabeled, no ceiling). They are
mapped to Uni3R's 8 classes [wall, floor, ceiling, chair, table, sofa, bed, other];
pixels outside the 20 classes stay 0 (ignored), so mIoU is only indicative.
"""
import json
import os
import os.path as osp

import numpy as np
import cv2
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

from .scannet import camera_normalization

# SIU3R semantic id -> Uni3R 8-class id (1-based, 0 = ignore)
_SIU3R_TO_UNI3R = np.full(256, 8, dtype=np.uint16)
_SIU3R_TO_UNI3R[0] = 0
_SIU3R_TO_UNI3R[1] = 1   # wall
_SIU3R_TO_UNI3R[2] = 2   # floor
_SIU3R_TO_UNI3R[5] = 4   # chair
_SIU3R_TO_UNI3R[7] = 5   # table
_SIU3R_TO_UNI3R[6] = 6   # sofa
_SIU3R_TO_UNI3R[4] = 7   # bed


class SIU3RTestDataset(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, pairs_file=None, scenes=None, num_views=2,
                 mask_bg=False, one_target_per_pair=False, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert num_views == 2, 'only the 2-view protocol is implemented'
        self.num_views = num_views
        self.mask_bg = mask_bg
        pairs_file = pairs_file or osp.join(ROOT, 'val_pair.json')
        with open(pairs_file) as f:
            pairs = json.load(f)
        available = {d for d in os.listdir(ROOT) if osp.isdir(osp.join(ROOT, d))}
        keep = set(scenes) if scenes else available
        self.all_views = []
        for p in pairs:
            scan = p['scan']
            if scan not in available or scan not in keep:
                continue
            c1, c2 = p['context_ids']
            targets = [t for t in p['target_ids'] if t not in (c1, c2)]
            if one_target_per_pair and targets:
                targets = [targets[len(targets) // 2]]  # deterministic middle target
            for t in targets:
                # same order as TestDataset: (target, source2, source1) popped from the right
                self.all_views.append((scan, (t, c2, c1)))
        self.scenes = sorted({v[0] for v in self.all_views})
        if not self.all_views:
            raise ValueError(f'No SIU3R pairs found under {ROOT}')

    def __len__(self):
        return len(self.all_views)

    def _load(self, scan, fid):
        root = osp.join(self.ROOT, scan)
        rgb = imread_cv2(osp.join(root, 'color', f'{fid}.jpg'))
        depth = imread_cv2(osp.join(root, 'depth', f'{fid}.png'), cv2.IMREAD_UNCHANGED)
        sem = imread_cv2(osp.join(root, 'semantic', f'{fid}.png'), cv2.IMREAD_UNCHANGED)
        K = np.loadtxt(osp.join(root, 'intrinsic.txt'))[:3, :3].astype(np.float32)
        c2w = np.loadtxt(osp.join(root, 'extrinsic', f'{fid}.txt')).astype(np.float32)
        return rgb, depth, sem, K, c2w

    def _get_views(self, idx, resolution, rng):
        scan, ids = self.all_views[idx]
        views = []
        for fid in reversed(ids):  # source1, source2, target
            rgb, depth, sem, K, c2w = self._load(scan, fid)
            if not np.isfinite(c2w).all():
                raise ValueError(f'invalid pose {scan}/{fid}')
            label = _SIU3R_TO_UNI3R[sem.astype(np.int64)]
            mask = np.ones_like(depth) * 255
            dml = np.stack([depth, mask, label.astype(depth.dtype)], axis=-1)
            rgb, dml, K = self._crop_resize_if_necessary(rgb, dml, K, resolution, rng=rng,
                                                         info=f'{scan}/{fid}')
            depthmap = dml[:, :, 0].astype(np.float32) / 1000
            views.append(dict(
                img=rgb, depthmap=depthmap, camera_pose=c2w, camera_intrinsics=K,
                extrinsics=np.copy(c2w), scale=np.float32(1.0), dataset='SIU3R',
                label=scan, instance=f'{fid}.jpg', labelmap=dml[:, :, 2].astype(np.int64)))
        scale = np.linalg.norm(views[0]['extrinsics'][:3, 3] - views[1]['extrinsics'][:3, 3])
        for i in range(len(views)):
            views[i]['extrinsics'][:3, 3] /= scale
            if i == 0:
                extrinsics0 = views[i]['extrinsics']
            views[i]['extrinsics'] = camera_normalization(extrinsics0, views[i]['extrinsics'])
            views[i]['scale'] = scale
        return views
