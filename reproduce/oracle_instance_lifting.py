"""Assign GT source-pixel instance IDs to frozen Uni3R Gaussians and diagnose geometry.

No training, clustering, Hungarian rematching, or target-view pose optimization.
The optional global Sim(3) uses context RGB-D correspondences only and is reported
separately from the original two-view baseline-normalized camera convention.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from oracle_math import (camera_in_prediction_frame, decode_instance_rgb,
                         fit_similarity, fixed_id_metrics, grouping_3d,
                         gt_visible_coverage, transform_points, unproject)


def read_view(root, frame_id):
    rgb = np.array(Image.open(root / 'color' / f'{frame_id}.jpg').convert('RGB'))
    depth = np.array(Image.open(root / 'depth' / f'{frame_id}.png')).astype(np.float32) / 1000
    instance = decode_instance_rgb(Image.open(root / 'instance' / f'{frame_id}.png').convert('RGB'))
    semantic = np.array(Image.open(root / 'semantic' / f'{frame_id}.png'))
    K = np.loadtxt(root / 'intrinsic.txt')[:3, :3].astype(np.float32)
    c2w = np.loadtxt(root / 'extrinsic' / f'{frame_id}.txt')
    if rgb.shape != (256, 256, 3) or depth.shape != (256, 256) or instance.shape != depth.shape:
        raise ValueError('Expected SIU3R jointly preprocessed 256-square data; do not resize labels independently')
    if not np.isfinite(c2w).all() or abs(np.linalg.det(c2w[:3, :3]) - 1) > 1e-3:
        raise ValueError('Invalid camera pose')
    # Validate RGB channel order and documented class offsets (18 things vs 20 semantic classes).
    labeled = instance > 0
    if np.any(labeled & (instance // 1000 + 2 != semantic)):
        raise ValueError('Instance encoding disagrees with semantic labels')
    return dict(id=int(frame_id), rgb=rgb, depth=depth, instance=instance,
                semantic=semantic, K=K, c2w=c2w, w2c=np.linalg.inv(c2w),
                gt_points=unproject(depth, K, c2w))


def colorize(ids):
    from colorsys import hsv_to_rgb
    result = np.full((*ids.shape, 3), 35, np.uint8)
    for k in np.unique(ids):
        if k <= 0:
            continue
        hue = ((int(k) * .61803398875) % 1)
        result[ids == k] = np.array(hsv_to_rgb(hue, .70, .95)) * 255
    return result


def panel(path, images, captions):
    width = 256
    canvas = Image.new('RGB', (len(images) * width, 282), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (img, caption) in enumerate(zip(images, captions)):
        canvas.paste(Image.fromarray(np.asarray(img, np.uint8)), (i * width, 26))
        draw.text((i * width + 5, 5), caption, fill='black')
    canvas.save(path)


def write_ply(path, points, labels, source):
    from plyfile import PlyData, PlyElement
    rgb = colorize(labels)
    v = np.empty(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                    ('instance_id', 'i4'), ('source_view', 'u1')])
    for j, key in enumerate(('x', 'y', 'z')):
        v[key] = points[:, j]
    for j, key in enumerate(('red', 'green', 'blue')):
        v[key] = rgb[:, j]
    v['instance_id'], v['source_view'] = labels, source
    PlyData([PlyElement.describe(v, 'vertex')], text=False).write(path)


def render_labels(pred, labels, source_indices, target, sim, all_ids, alpha_threshold):
    from gsplat import rasterization
    chosen = [pred[j] for j in source_indices]
    device = chosen[0]['means'].device
    means = torch.cat([p['means'].reshape(-1, 3) for p in chosen])
    quats = torch.cat([p['rotations'].reshape(-1, 4) for p in chosen])
    scales = torch.cat([p['scales'].reshape(-1, 3) for p in chosen])
    opacity = torch.cat([p['opacities'].reshape(-1) for p in chosen])
    native_ids = np.concatenate([labels[j].ravel() for j in source_indices])
    lookup = np.searchsorted(all_ids, native_ids)
    onehot = torch.nn.functional.one_hot(torch.as_tensor(lookup, device=device), len(all_ids)).float()
    camera = camera_in_prediction_frame(target['w2c'], sim)
    features, alpha, _ = rasterization(
        means, quats, scales, opacity, onehot,
        torch.as_tensor(camera, dtype=torch.float32, device=device)[None],
        torch.as_tensor(target['K'], device=device)[None], width=256, height=256,
        near_plane=.1 / sim[0], far_plane=100 / sim[0], render_mode='RGB+ED', channel_chunk=32)
    alpha = alpha[0, ..., 0].cpu().numpy()
    weights = features[0, ..., :len(all_ids)].cpu().numpy()
    result = all_ids[weights.argmax(-1)].copy()
    result[alpha < alpha_threshold] = 0
    # Expected depth is in native prediction units; convert to metric depth.
    depth = features[0, ..., -1].cpu().numpy() * sim[0]
    return result, alpha, depth


def reprojection_control(sources, target, all_ids):
    """GT RGB-D point reprojection control, not a second learned model."""
    points = np.concatenate([s['gt_points'][s['depth'] > 0] for s in sources])
    labels = np.concatenate([s['instance'][s['depth'] > 0] for s in sources])
    p = points @ target['w2c'][:3, :3].T + target['w2c'][:3, 3]
    keep = np.isfinite(p).all(-1) & (p[:, 2] > .01)
    p, labels = p[keep], labels[keep]
    uvh = p @ target['K'].T
    uv = np.rint(uvh[:, :2] / uvh[:, 2:]).astype(int)
    # Fixed 3x3 point footprint for discrete raster sampling.
    zbuffer = np.full((256, 256), np.inf)
    output = np.zeros((256, 256), np.int64)
    for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        u, v = uv[:, 0] + dx, uv[:, 1] + dy
        ok = (u >= 0) & (u < 256) & (v >= 0) & (v < 256)
        u, v, z, lab = u[ok], v[ok], p[ok, 2], labels[ok]
        # Unique closest candidate per pixel, then combine footprints by z-buffer.
        order = np.argsort(z)
        linear = v[order] * 256 + u[order]
        _, first = np.unique(linear, return_index=True)
        idx = order[first]
        u, v, z, lab = u[idx], v[idx], z[idx], lab[idx]
        win = z < zbuffer[v, u]
        zbuffer[v[win], u[win]] = z[win]
        output[v[win], u[win]] = lab[win]
    valid = (target['semantic'] > 0) & (target['depth'] > 0)
    return fixed_id_metrics(output, target['instance'], valid, all_ids[all_ids > 0])


def evaluate_pair(model, root, pair, output, args):
    name = f"{pair['scan']}_{pair['context_ids'][0]}_{pair['context_ids'][1]}"
    dest = output / name
    dest.mkdir(parents=True, exist_ok=True)
    frames = sorted(set(pair['context_ids'] + pair['target_ids']))
    views = {i: read_view(root / pair['scan'], i) for i in frames}
    source = [views[i] for i in pair['context_ids']]
    context = [{'img': torch.from_numpy(s['rgb'].copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1,
                'camera_intrinsics': torch.from_numpy(s['K'].copy())[None].cuda()} for s in source]
    torch.cuda.synchronize()
    start = time.perf_counter()
    pred = model(context)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    labels = [s['instance'] for s in source]
    all_ids, counts = np.unique(np.concatenate([x.ravel() for x in labels]), return_counts=True)
    if 0 not in all_ids:
        all_ids = np.insert(all_ids, 0, 0)
    eligible = np.array([k for k in all_ids if k > 0 and sum((x == k).sum() for x in labels) >= 64])
    xyz = np.stack([p['means'][0].cpu().numpy() for p in pred])
    source_gt = np.stack([s['gt_points'] for s in source])
    source_valid = np.stack([s['depth'] > 0 for s in source])
    baseline = np.linalg.norm(source[0]['c2w'][:3, 3] - source[1]['c2w'][:3, 3])
    if baseline < 1e-4:
        raise ValueError('Degenerate input baseline')
    native = (baseline, source[0]['c2w'][:3, :3], source[0]['c2w'][:3, 3])
    # Hold every fourth pixel out from similarity fitting for diagnostic residuals.
    yy, xx = np.indices((256, 256))
    fit_mask = source_valid & (((xx + yy) % 4) != 0)[None]
    aligned = fit_similarity(xyz[fit_mask], source_gt[fit_mask])
    simulations = {'baseline_gauge': native, 'context_sim3': aligned}
    result = {'name': name, 'scene': pair['scan'], 'context_ids': pair['context_ids'],
              'official_pair_index': pair['original_pair_index'], 'published_context_overlap': pair['iou'],
              'source_object_ids': all_ids.tolist(), 'eligible_object_ids': eligible.tolist(),
              'native_gaussian_count': int(xyz.size // 3), 'baseline_m': float(baseline),
              'forward_seconds_including_original_teachers': seconds, 'renders': [], 'geometry': {}}
    gt_cloud = np.concatenate([v['gt_points'][v['depth'] > 0] for v in views.values()])
    gt_ids = np.concatenate([v['instance'][v['depth'] > 0] for v in views.values()])
    flat_labels = np.concatenate([x.ravel() for x in labels])
    opacity = np.concatenate([p['opacities'].cpu().numpy().ravel() for p in pred])
    export = np.isfinite(xyz.reshape(-1, 3)).all(-1) & (opacity >= .05) & (flat_labels > 0)
    src_ids = np.repeat(np.arange(2), 256 * 256)
    for mode, sim in simulations.items():
        world = transform_points(xyz, sim)
        test_mask = source_valid & ~fit_mask
        residual = np.linalg.norm(world[test_mask] - source_gt[test_mask], axis=-1)
        result['geometry'][mode] = {
            'scale': sim[0], 'rotation': sim[1].tolist(), 'translation': sim[2].tolist(),
            'heldout_context_correspondence_median_m': float(np.median(residual)),
            'heldout_context_correspondence_p90_m': float(np.quantile(residual, .9)),
            'grouping': grouping_3d(world.reshape(-1, 3)[export], flat_labels[export], gt_cloud, gt_ids)}
        if mode == 'context_sim3':
            write_ply(dest / 'instances_context_sim3.ply', world.reshape(-1, 3)[export], flat_labels[export], src_ids[export])
    np.savez_compressed(dest / 'gaussians_native.npz', means=xyz, instance_ids=np.stack(labels),
                        scales=np.stack([p['scales'][0].cpu().numpy() for p in pred]),
                        rotations=np.stack([p['rotations'][0].cpu().numpy() for p in pred]),
                        opacity=np.stack([p['opacities'][0].cpu().numpy() for p in pred]))
    panel(dest / 'sources.png', [source[0]['rgb'], colorize(labels[0]), source[1]['rgb'], colorize(labels[1])],
          [f"Source {source[0]['id']}", 'GT instance IDs', f"Source {source[1]['id']}", 'GT instance IDs'])
    jobs = [('cross_source', source[0]['id'], [1]), ('cross_source', source[1]['id'], [0])]
    jobs += [('novel_view', i, [0, 1]) for i in pair['target_ids'] if i not in pair['context_ids']]
    for kind, fid, selected_sources in jobs:
        target = views[fid]
        valid = (target['semantic'] > 0) & (target['depth'] > 0)
        gt_source_points = np.concatenate([source[j]['gt_points'][source[j]['depth'] > 0] for j in selected_sources])
        coverage = gt_visible_coverage(gt_source_points, target['depth'], target['K'], target['w2c'])
        selected_ids = np.unique(np.concatenate([labels[j].ravel() for j in selected_sources]))
        ids = np.intersect1d(eligible, selected_ids)
        record = {'kind': kind, 'target_id': fid, 'source_indices': selected_sources,
                  'gt_coverage_fraction': float(coverage[valid].mean()) if valid.any() else None,
                  'gt_depth_reprojection_control': reprojection_control([source[j] for j in selected_sources], target, np.r_[0, ids]),
                  'modes': {}}
        rendered = {}
        for mode, sim in simulations.items():
            mask, alpha, depth = render_labels(pred, labels, selected_sources, target, sim, all_ids, args.alpha_threshold)
            rendered[mode] = mask
            scored = fixed_id_metrics(mask, target['instance'], valid, ids)
            covered = fixed_id_metrics(mask, target['instance'], valid & coverage, ids)
            target_seen = valid & np.isin(target['instance'], ids)
            metric_depth = valid & (alpha >= args.alpha_threshold) & np.isfinite(depth) & (depth > 0)
            depth_rel = np.abs(depth[metric_depth] - target['depth'][metric_depth]) / target['depth'][metric_depth]
            record['modes'][mode] = {'all_valid': scored, 'gt_covered': covered,
                                    'render_alpha_coverage': float((alpha[valid] >= args.alpha_threshold).mean()),
                                    'foreground_uncovered_fraction': float((alpha[target_seen] < args.alpha_threshold).mean()) if target_seen.any() else None,
                                    'depth_absrel_no_per_view_alignment': float(depth_rel.mean()) if depth_rel.size else None}
            np.savez_compressed(dest / f'{kind}_{fid}_{mode}.npz', instance=mask.astype(np.int32),
                                alpha=alpha, depth=depth, gt_instance=target['instance'].astype(np.int32),
                                valid=valid, gt_coverage=coverage)
        err = np.full((256, 256, 3), 35, np.uint8)
        focus = valid & np.isin(target['instance'], ids)
        correct = rendered['context_sim3'] == target['instance']
        err[focus & correct] = [55, 190, 85]
        err[focus & ~correct] = [230, 60, 50]
        panel(dest / f'{kind}_{fid}.png',
              [target['rgb'], colorize(target['instance']), colorize(rendered['baseline_gauge']),
               colorize(rendered['context_sim3']), err],
              [f'{kind} target {fid}', 'GT instance IDs', 'Baseline gauge (no opt)', 'One context-only Sim3', 'Seen GT: green=correct'])
        result['renders'].append(record)
    (dest / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'pair': name, 'objects': len(eligible),
                      'novel_miou': {m: np.mean([r['modes'][m]['all_valid']['mean_iou'] for r in result['renders']
                                               if r['kind'] == 'novel_view' and r['modes'][m]['all_valid']['mean_iou'] is not None])
                                    for m in simulations}}), flush=True)
    del pred
    return result


def build_report(output, results):
    lines = ['# GT instance → Uni3R Gaussian diagnostic', '',
             'Official frozen 2-view checkpoint. No model training or target-view pose optimization.',
             'GT labels are directly attached by source-pixel index. This is an oracle-label geometry diagnostic, not learned segmentation.', '',
             '| Scene / input pair | Context overlap | Novel IoU baseline gauge | Novel IoU context Sim3 | GT-depth point control IoU |',
             '|---|---:|---:|---:|---:|']
    table = []
    for p in results:
        novel = [r for r in p['renders'] if r['kind'] == 'novel_view']
        mean = lambda vs: float(np.mean([v for v in vs if v is not None])) if any(v is not None for v in vs) else None
        row = {'pair': p['name'], 'scene': p['scene'], 'context_overlap': p['published_context_overlap']}
        for mode in ('baseline_gauge', 'context_sim3'):
            row[mode] = mean([r['modes'][mode]['all_valid']['mean_iou'] for r in novel])
            row[mode + '_covered'] = mean([r['modes'][mode]['gt_covered']['mean_iou'] for r in novel])
        row['gt_point_control'] = mean([r['gt_depth_reprojection_control']['mean_iou'] for r in novel])
        table.append(row)
        fmt = lambda x: 'N/A' if x is None else f'{x:.4f}'
        lines.append(f"| {p['name']} | {row['context_overlap']:.3f} | {fmt(row['baseline_gauge'])} | {fmt(row['context_sim3'])} | {fmt(row['gt_point_control'])} |")
    lines += ['', 'Instance IoU uses original GT IDs, no matching or merging; objects must have >=64 source pixels and >=32 valid target pixels.',
              'Valid pixels: labeled semantic regions with positive GT depth. Unseen target objects are not counted as evaluable source instances.',
              'All-valid scores include uncovered target surfaces; GT-covered scores use only GT depth reprojection (3x3 footprint, 3cm+2% depth tolerance).',
              'Baseline gauge: first source GT camera + input GT camera baseline, matching the original two-view coordinate convention.',
              'Context Sim3: one robust similarity fitted from source RGB-D correspondences only; excludes every fourth pixel for residual evaluation. Privileged diagnostic, not deployment.',
              'The GT-depth control uses point z-buffering, not learned Gaussian attributes, so it is a calibration/reference rather than a strict upper bound.',
              '3D metrics use observed GT depth surfaces, not a complete scan mesh. They are not official ScanNet instance AP.',
              'Camera intrinsics, GT masks, RGB and depth use the SIU3R jointly preprocessed 256x256 files. Frames differ from the original Uni3R test protocol.',
              'Thresholds were fixed before inference; scenes selected by small archive size among overlapping scene IDs, not by output quality.', '',
              'Open index.html for all source/target panels. Each pair has metrics.json, an instance-colored PLY, and native Gaussian geometry with IDs in NPZ.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    (output / 'summary.json').write_text(json.dumps(table, indent=2, allow_nan=False))
    html = ['<!doctype html><meta charset="utf-8"><title>GT instance lifting diagnostic</title>',
            '<style>body{font:16px sans-serif;background:#eee;margin:24px}img{max-width:100%;height:auto}section{background:white;padding:16px;margin:20px 0}</style>',
            '<h1>GT instance → frozen Uni3R</h1><p>No training, clustering, or target pose optimization. Green = correct GT identity; red = incorrect. Dark = unscored.</p>',
            '<p>Baseline gauge uses source GT camera/baseline; Sim3 uses context RGB-D only. These are geometry diagnostics, not instance prediction benchmarks.</p>']
    for p in results:
        html += [f"<section><h2>{p['name']}</h2><a href='{p['name']}/metrics.json'>metrics</a> · <a href='{p['name']}/instances_context_sim3.ply'>3D instances PLY</a>",
                 f"<p><img loading='lazy' src='{p['name']}/sources.png'></p>"]
        html += [f"<p><img loading='lazy' src='{p['name']}/{r['kind']}_{r['target_id']}.png'></p>" for r in p['renders']]
        html.append('</section>')
    (output / 'index.html').write_text('\n'.join(html))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, default=REPO / 'outputs/instance_oracle/overlap_data')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/instance_oracle/run')
    parser.add_argument('--checkpoint', type=Path, default=Path('/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R/checkpoints/2views.pth'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--alpha-threshold', type=float, default=.5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from uni3r.utils.path_manager import init_all_submodules
    init_all_submodules()
    from uni3r.uni3r import Uni3R
    torch.manual_seed(777)
    np.random.seed(777)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = json.loads((args.data / 'pairs.json').read_text())
    if args.limit:
        pairs = pairs[:args.limit]
    manifest = {'checkpoint': str(args.checkpoint), 'seed': 777, 'input_views': 2, 'amp': False,
                'pose_optimization_steps': 0, 'alpha_threshold': args.alpha_threshold,
                'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), REPO / 'reproduce/oracle_math.py', REPO / 'uni3r/uni3r.py', REPO / 'uni3r/gaussian_head.py']},
                'data_provenance': json.loads((args.data / 'provenance.json').read_text()),
                'note': 'Original full Uni3R forward retained including teachers; GT labels replace only semantic render channels.'}
    previous = args.output / 'manifest.json'
    if args.resume and previous.exists():
        old = json.loads(previous.read_text())
        assert old == manifest, 'Refuse resume across changed source/config/data'
    previous.write_text(json.dumps(manifest, indent=2))
    print('Loading original Uni3R 2-view checkpoint', flush=True)
    model = Uni3R.from_pretrained(str(args.checkpoint), device='cuda', num_views_value=2).eval().requires_grad_(False)
    results = []
    with torch.no_grad():
        for pair in pairs:
            name = f"{pair['scan']}_{pair['context_ids'][0]}_{pair['context_ids'][1]}"
            existing = args.output / name / 'metrics.json'
            if args.resume and existing.exists():
                result = json.loads(existing.read_text())
            else:
                result = evaluate_pair(model, args.data, pair, args.output, args)
            results.append(result)
            build_report(args.output, results)
    print('COMPLETED', len(results), 'pairs', args.output, flush=True)


if __name__ == '__main__':
    main()
