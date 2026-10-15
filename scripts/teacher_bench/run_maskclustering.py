"""MaskClustering (CVPR'24, official code external/MaskClustering @ eb2d41c, unmodified) on the
teacher-benchmark scans, then projected back to 2D context frames.

Adaptations (only I/O, algorithm untouched):
  * 2D masks: SAM2.1-L AMG (work/sam2_frames) instead of CropFormer (weights offline).
  * Scene point cloud: SIU3R data ships no mesh, so we fuse the GT-depth back-projections of
    all used frames, voxel-downsampled to 2 cm (MaskClustering uses the mesh vertices).
  * Frames: every 10th frame (official step=10) + the benchmark context frames.
  * pytorch3d.ops.ball_query -> exact torch implementation (mc_shim/), avoids a CUDA build.
  * Config: official configs/scannet.json thresholds.
Back to 2D: each 3D object keeps the list of 2D masks merged into it (object_dict mask_list);
for a context frame we paint every mask of that frame belonging to object k with id k+1.
Score = number of supporting masks (view support), so AP ranks by multi-view consensus.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'mc_shim'))
MC = os.environ.get('MASKCLUSTERING_ROOT', os.path.abspath(os.path.join(HERE, '../../external/MaskClustering')))
sys.path.insert(0, MC)
from graph.construction import mask_graph_construction  # noqa: E402
from graph.iterative_clustering import iterative_clustering  # noqa: E402
import utils.post_process as pp  # noqa: E402

DATA = os.environ.get('TEACHER_BENCH_DATA', os.path.abspath(os.path.join(HERE, '../../.cache/teacher_bench_data')))


class BenchDataset:
    def __init__(self, scan, seg_dir):
        self.root = f'{DATA}/{scan}'
        self.seg_dir = seg_dir
        K = np.loadtxt(f'{self.root}/intrinsic.txt')
        self.K = K
        self.image_size = (256, 256)
        self.cam = o3d.camera.PinholeCameraIntrinsic()
        self.cam.set_intrinsics(256, 256, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    def get_intrinsics(self, fid):
        return self.cam

    def get_extrinsic(self, fid):
        return np.loadtxt(f'{self.root}/extrinsic/{fid}.txt')

    def get_depth(self, fid):
        return (np.array(Image.open(f'{self.root}/depth/{fid}.png')) / 1000.0).astype(np.float32)

    def get_segmentation(self, fid, align_with_depth=False):
        return np.array(Image.open(f'{self.seg_dir}/{fid}.png')).astype(np.int64)

    def frame_list(self):
        return sorted(int(f[:-4]) for f in os.listdir(f'{self.root}/depth'))

    def scene_points(self, frames):
        pts = []
        for f in frames:
            E = self.get_extrinsic(f)
            if not np.isfinite(E).all():
                continue
            pc = o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(self.get_depth(f)), self.cam, depth_scale=1, depth_trunc=20)
            pc.transform(E)
            pts.append(np.asarray(pc.points))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.concatenate(pts))
        return np.asarray(pc.voxel_down_sample(0.02).points)


def cluster_scan(scan, work):
    ds = BenchDataset(scan, f'{work}/sam2_frames/{scan}')
    frames = ds.frame_list()
    cfg = json.load(open(f'{MC}/configs/scannet.json'))
    args = SimpleNamespace(**cfg, debug=False)
    pts = ds.scene_points(frames)
    with torch.no_grad():
        nodes, thr, mask_pcs, pfm = mask_graph_construction(args, pts, frames, ds)
        objs = iterative_clustering(nodes, thr, args.view_consensus_threshold, False)
    # official post-process (DBSCAN split, point filter, overlap merge) without file export
    ids_l, bbox_l, mask_l = [], [], []
    for node in objs:
        if len(node.mask_list) < 2:
            continue
        pcld, pids = node.get_point_cloud(pts)
        pl, il = pp.dbscan_process(pcld, pids)
        a, b, c = pp.filter_point(pfm, node, pl, il, mask_pcs, frames, args)
        ids_l += a; bbox_l += b; mask_l += c
    ids_l, mask_l = pp.merge_overlapping_objects(ids_l, bbox_l, mask_l, overlapping_ratio=0.8)
    return mask_l, len(pts), len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default=f'{DATA}/pairs.json')
    ap.add_argument('--work', default='/opt/jensen_cache/teacher_bench/work')
    ap.add_argument('--out', default='/opt/jensen_cache/teacher_bench/maskclustering')
    ap.add_argument('--shard_count', type=int, default=1)
    ap.add_argument('--shard_index', type=int, default=0)
    args = ap.parse_args()
    pairs = json.load(open(args.pairs))
    scans = sorted({p['scan'] for p in pairs})[args.shard_index::args.shard_count]
    for scan in scans:
        todo = [p for p in pairs if p['scan'] == scan and
                not os.path.exists(f"{args.out}/{scan}_{p['context_ids'][0]}_{p['context_ids'][1]}/meta.json")]
        if not todo:
            continue
        objects, npts, nfr = cluster_scan(scan, args.work)
        seg_dir = f'{args.work}/sam2_frames/{scan}'
        for p in todo:
            c1, c2 = p['context_ids']
            frames, meta = {}, {}
            for fid in (c1, c2):
                seg = np.array(Image.open(f'{seg_dir}/{fid}.png')).astype(np.int64)
                idmap = np.zeros(seg.shape, np.uint16)
                # paint larger objects first so small ones stay on top
                order = sorted(range(len(objects)), key=lambda k: -len(objects[k]))
                for k in order:
                    for f, mid, _cov in objects[k]:
                        if f == fid:
                            idmap[seg == mid] = k + 1
                frames[fid] = idmap
            for k in range(len(objects)):
                if any((frames[f] == k + 1).any() for f in frames):
                    meta[str(k + 1)] = {'score': float(len(objects[k])), 'label': None}
            d = f'{args.out}/{scan}_{c1}_{c2}'
            os.makedirs(d, exist_ok=True)
            for f, m in frames.items():
                Image.fromarray(m).save(f'{d}/{f}.png')
            json.dump(meta, open(f'{d}/meta.json', 'w'))
        print(f'done {scan}: {len(objects)} objects, {npts} pts, {nfr} frames', flush=True)


if __name__ == '__main__':
    main()
