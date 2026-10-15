"""Score teacher outputs on the validation pairs against SIU3R 2D instance GT.

Teacher output layout (all teachers):
  <out>/<scan>_<c1>_<c2>/<frame>.png   uint16 instance-id map (0 = no instance), 256x256
  <out>/<scan>_<c1>_<c2>/meta.json     {"<id>": {"score": float, "label": str|null}}
IDs are shared across the two frames of a pair iff the teacher claims cross-view identity.

GT: SIU3R instance png, id = R + 256G + 65536B = 1000*class + inst (class 1..18 = things);
id 0 = wall/floor/unlabeled -> ignore region. A predicted mask with > 50% of its area in
the ignore region is removed before scoring (neither TP nor FP). GT objects < 64 px ignored
for mCov / consistency (kept for AP, as COCO does).

Metrics:
  frame AP / AP50 / AR100  class-agnostic COCO mask AP, each frame one image
  pair  AP / AP50          the two frames stacked as ONE image (SIU3R global-ID style):
                           a teacher that changes ID across views loses here
  mCov                     mean over GT objects of the best IoU with any predicted mask
  consistency              among GT objects visible (>=64 px) in both views and matched
                           (IoU>=0.5) in both, fraction whose matched pred IDs are equal
  class AP50               only for teachers emitting ScanNet labels
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision

GT_ROOT = os.environ.get('TEACHER_BENCH_DATA', '/opt/jensen_cache/teacher_bench_data')
THINGS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
          'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
          'toilet', 'sink', 'bathtub', 'otherfurniture']  # SIU3R instance class 1..18


def load_gt(scan, fid):
    a = np.array(Image.open(f'{GT_ROOT}/{scan}/instance/{fid}.png').convert('RGB')).astype(np.int64)
    return a[..., 0] + 256 * a[..., 1] + 65536 * a[..., 2]


def masks_from_ids(idmap, ids):
    return np.stack([idmap == i for i in ids]) if len(ids) else np.zeros((0, *idmap.shape), bool)


def pred_instances(pred, ignore, meta):
    ids = [int(i) for i in np.unique(pred) if i > 0]
    keep, masks = [], []
    for i in ids:
        m = pred == i
        if (m & ignore).sum() > 0.5 * m.sum():
            continue
        keep.append(i)
        masks.append(m)
    scores = [float(meta.get(str(i), {}).get('score', 1.0)) for i in keep]
    labels = [meta.get(str(i), {}).get('label') for i in keep]
    return keep, (np.stack(masks) if masks else np.zeros((0, *pred.shape), bool)), scores, labels


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a.reshape(len(a), -1).astype(np.float32)
    b = b.reshape(len(b), -1).astype(np.float32)
    inter = a @ b.T
    return inter / (a.sum(1)[:, None] + b.sum(1)[None] - inter + 1e-6)


def tm_entry(masks, scores=None, labels=None):
    d = {'masks': torch.from_numpy(masks).bool(),
         'labels': torch.tensor(labels if labels is not None else [0] * len(masks), dtype=torch.long)}
    if scores is not None:
        d['scores'] = torch.tensor(scores, dtype=torch.float)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='/opt/jensen_cache/teacher_bench_data/pairs.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    pairs = json.load(open(args.pairs))
    frame_ap = MeanAveragePrecision(iou_type='segm', max_detection_thresholds=[1, 10, 100])
    pair_ap = MeanAveragePrecision(iou_type='segm', max_detection_thresholds=[1, 10, 100])
    class_ap = MeanAveragePrecision(iou_type='segm', iou_thresholds=[0.5])
    covs, cons_num, cons_den, both_vis, n_pred, missing = [], 0, 0, 0, [], 0
    has_labels = False
    for p in pairs:
        scan, (c1, c2) = p['scan'], p['context_ids']
        d = f"{args.out}/{scan}_{c1}_{c2}"
        if not os.path.exists(f'{d}/meta.json'):
            missing += 1
            continue
        meta = json.load(open(f'{d}/meta.json'))
        per_view = []
        for fid in (c1, c2):
            gt = load_gt(scan, fid)
            pred = np.array(Image.open(f'{d}/{fid}.png')).astype(np.int64)
            assert pred.shape == gt.shape, (pred.shape, gt.shape)
            ignore = gt == 0
            gids = [int(i) for i in np.unique(gt) if i > 0]
            gmasks = masks_from_ids(gt, gids)
            pids, pmasks, pscores, plabels = pred_instances(pred, ignore, meta)
            n_pred.append(len(pids))
            frame_ap.update([tm_entry(pmasks, pscores)], [tm_entry(gmasks)])
            if any(l is not None for l in plabels):
                has_labels = True
                ok = [k for k, l in enumerate(plabels) if l in THINGS]
                class_ap.update([tm_entry(pmasks[ok], [pscores[k] for k in ok],
                                          [THINGS.index(plabels[k]) for k in ok])],
                                [tm_entry(gmasks, labels=[g // 1000 - 1 for g in gids])])
            M = iou_matrix(gmasks, pmasks)
            for gi, g in enumerate(gids):
                if gmasks[gi].sum() >= 64:
                    covs.append(M[gi].max() if M.shape[1] else 0.0)
            per_view.append((gt, pred, ignore, gids, gmasks, pids, pmasks, pscores, M))
        # pair-level (stacked) AP with the teacher's own IDs
        gt_s = np.concatenate([per_view[0][0], per_view[1][0]], 0)
        pr_s = np.concatenate([per_view[0][1], per_view[1][1]], 0)
        gids_s = [int(i) for i in np.unique(gt_s) if i > 0]
        pids_s, pmasks_s, pscores_s, _ = pred_instances(pr_s, gt_s == 0, meta)
        pair_ap.update([tm_entry(pmasks_s, pscores_s)], [tm_entry(masks_from_ids(gt_s, gids_s))])
        # cross-view identity consistency
        (_, _, _, g0, gm0, p0, _, _, M0), (_, _, _, g1, gm1, p1, _, _, M1) = per_view
        for g in set(g0) & set(g1):
            i0, i1 = g0.index(g), g1.index(g)
            if gm0[i0].sum() < 64 or gm1[i1].sum() < 64:
                continue
            both_vis += 1
            if M0.shape[1] == 0 or M1.shape[1] == 0:
                continue
            j0, j1 = M0[i0].argmax(), M1[i1].argmax()
            if M0[i0, j0] >= 0.5 and M1[i1, j1] >= 0.5:
                cons_den += 1
                cons_num += int(p0[j0] == p1[j1])
    fa, pa = frame_ap.compute(), pair_ap.compute()
    res = {
        'pairs_scored': len(pairs) - missing, 'pairs_missing': missing,
        'frame_AP': float(fa['map']), 'frame_AP50': float(fa['map_50']), 'frame_AR100': float(fa['mar_100']),
        'pair_AP': float(pa['map']), 'pair_AP50': float(pa['map_50']),
        'mCov': float(np.mean(covs)), 'gt_objects': len(covs),
        'consistency': cons_num / max(cons_den, 1), 'consistency_matched': cons_den,
        'both_view_objects': both_vis, 'preds_per_frame': float(np.mean(n_pred)),
    }
    if has_labels:
        res['class_AP50'] = float(class_ap.compute()['map'])
    print(json.dumps(res, indent=1))
    json.dump(res, open(f'{args.out}/scores.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
