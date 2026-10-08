"""Geometry and fixed-ID metrics for the GT-instance lifting diagnostic."""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree


def decode_instance_rgb(rgb):
    """SIU3R/COCO RGB packed ID; retain class*1000+instance, not class alone."""
    rgb = np.asarray(rgb, dtype=np.int64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected an RGB-packed GT instance image')
    return rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]


def unproject(depth, K, c2w):
    yy, xx = np.indices(depth.shape)
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(K).T
    camera = rays * depth[..., None]
    return camera @ c2w[:3, :3].T + c2w[:3, 3]


def fit_similarity(source, target, trim=0.8, iterations=4):
    """One robust scene-wide source->GT Sim(3), never per-view alignment."""
    x, y = np.asarray(source, np.float64), np.asarray(target, np.float64)
    ok = np.isfinite(x).all(-1) & np.isfinite(y).all(-1)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        raise ValueError('Too few valid context points for alignment')
    keep = np.ones(len(x), bool)
    for _ in range(iterations):
        a, b = x[keep], y[keep]
        ma, mb = a.mean(0), b.mean(0)
        ac, bc = a - ma, b - mb
        u, d, vt = np.linalg.svd(bc.T @ ac / len(a))
        sign = np.ones(3)
        sign[-1] = np.sign(np.linalg.det(u @ vt))
        R = (u * sign) @ vt
        variance = np.mean(np.sum(ac * ac, axis=-1))
        if variance < 1e-12:
            raise ValueError('Degenerate alignment points')
        scale = np.sum(d * sign) / variance
        translation = mb - scale * (R @ ma)
        err = np.linalg.norm(scale * (x @ R.T) + translation - y, axis=-1)
        keep = err <= np.quantile(err, trim)
    return float(scale), R, translation


def transform_points(x, sim):
    s, R, t = sim
    return s * (x @ R.T) + t


def camera_in_prediction_frame(gt_w2c, sim):
    """World'=s R World+t. Transform camera, retaining native GS attributes."""
    s, R, t = sim
    out = np.eye(4)
    out[:3, :3] = gt_w2c[:3, :3] @ R
    out[:3, 3] = (gt_w2c[:3, :3] @ t + gt_w2c[:3, 3]) / s
    return out


def gt_visible_coverage(source_points, target_depth, K, w2c):
    """Coverage from GT geometry only, with an explicit occlusion tolerance."""
    h, w = target_depth.shape
    p = source_points @ w2c[:3, :3].T + w2c[:3, 3]
    front = np.isfinite(p).all(-1) & (p[:, 2] > 0.01)
    p = p[front]
    uvh = p @ K.T
    uv = np.rint(uvh[:, :2] / uvh[:, 2:]).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, z = uv[inside], p[inside, 2]
    gt = target_depth[uv[:, 1], uv[:, 0]]
    visible = (gt > 0) & (np.abs(gt - z) < 0.03 + 0.02 * gt)
    mask = np.zeros((h, w), bool)
    mask[uv[visible, 1], uv[visible, 0]] = True
    # One-pixel dilation accounts for discrete source sampling, not predictions.
    return maximum_filter(mask, size=3) & (target_depth > 0)


def fixed_id_metrics(pred, gt, valid, object_ids, min_pixels=32):
    """No Hungarian matching: an incorrect GT object identity stays incorrect."""
    records = []
    for k in object_ids:
        g = (gt == k) & valid
        if g.sum() < min_pixels:
            continue
        p = (pred == k) & valid
        intersection = int((g & p).sum())
        union = int((g | p).sum())
        records.append({'id': int(k), 'gt_pixels': int(g.sum()), 'pred_pixels': int(p.sum()),
                        'intersection': intersection, 'union': union, 'iou': intersection / union})
    foreground = valid & np.isin(gt, object_ids)
    return {'instances': records, 'num_instances': len(records),
            'mean_iou': float(np.mean([r['iou'] for r in records])) if records else None,
            'foreground_accuracy': float((pred[foreground] == gt[foreground]).mean()) if foreground.any() else None,
            'valid_pixels': int(valid.sum()), 'foreground_pixels': int(foreground.sum())}


def grouping_3d(pred_points, pred_ids, gt_points, gt_ids):
    """Observed-surface geometry and fixed-ID purity, not official scene AP."""
    valid = np.isfinite(pred_points).all(-1) & (pred_ids > 0)
    pred_points, pred_ids = pred_points[valid], pred_ids[valid]
    valid = np.isfinite(gt_points).all(-1) & (gt_ids > 0)
    gt_points, gt_ids = gt_points[valid], gt_ids[valid]
    if not len(pred_points) or not len(gt_points):
        return {'num_pred_points': len(pred_points), 'num_gt_points': len(gt_points)}
    d, indices = cKDTree(gt_points).query(pred_points, workers=2)
    same_id = pred_ids == gt_ids[indices]
    per_object = []
    for k in np.intersect1d(np.unique(pred_ids), np.unique(gt_ids)):
        p, g = pred_points[pred_ids == k], gt_points[gt_ids == k]
        if len(p) < 32 or len(g) < 32:
            continue
        dp = cKDTree(g).query(p, workers=2)[0]
        dg = cKDTree(p).query(g, workers=2)[0]
        precision, recall = (dp < .05).mean(), (dg < .05).mean()
        per_object.append({'id': int(k), 'pred_points': len(p), 'gt_points': len(g),
                           'accuracy_mean_m': float(dp.mean()), 'completeness_mean_m': float(dg.mean()),
                           'fscore_5cm': float(2 * precision * recall / max(precision + recall, 1e-12))})
    close = d < .05
    return {'num_pred_points': len(pred_points), 'num_gt_points': len(gt_points),
            'nearest_surface_median_m': float(np.median(d)),
            'within_5cm_fraction': float(close.mean()),
            'id_purity_within_5cm': float(same_id[close].mean()) if close.any() else None,
            'correct_id_and_within_5cm_fraction': float((same_id & close).mean()),
            'mean_object_fscore_5cm': float(np.mean([r['fscore_5cm'] for r in per_object])) if per_object else None,
            'per_object': per_object}
