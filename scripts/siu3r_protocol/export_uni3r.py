"""Run Uni3R on the SIU3R ScanNet val protocol and write outputs in SIU3R's format.

Faithfulness choices (see test_results/siu3r_protocol/README.md):
  * Data, GT masks and all output files come from SIU3R's own code (ScanNetDataset,
    Visualizer.visualize_recon_image / save_seg_ids / save_gt_seg_masks); scores come
    from SIU3R's Evaluator (evaluate_siu3r.py).
  * Inputs are SIU3R's 256x256 images and intrinsics as-is (no Uni3R crop), so renders
    are pixel-aligned with SIU3R's GT.
  * Target cameras: SIU3R's GT poses relative to the first context view, translation
    divided by the context baseline (Uni3R's native convention). No test-time pose
    optimisation (SIU3R does none either).
  * Semantics: LSeg decoding of the rendered feature field against the 20 ScanNet class
    names, every pixel labelled (no GT masking). Uni3R has no instance output, so
    instance ids are 0 and mAP/PQ are not evaluated (as SIU3R does for LSM).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from uni3r.utils.path_manager import init_all_submodules  # noqa: E402
init_all_submodules()
from uni3r.uni3r import Uni3R  # noqa: E402
from uni3r.loss import merge_and_split_predictions  # noqa: E402
from uni3r.utils.camera_utils import get_scaled_camera  # noqa: E402
from uni3r.utils.cuda_splatting import DummyPipeline, render  # noqa: E402
from uni3r.utils.gaussian_model import GaussianModel  # noqa: E402

# SIU3R official names for semantic ids 1..20 (0 = unlabeled)
SIU3R_CLASSES = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                 'otherfurniture']
# text prompts for LSeg (only change: split the compound word)
PROMPTS = [c if c != 'otherfurniture' else 'other furniture' for c in SIU3R_CLASSES]


def align_pose(gaussians, cam, K, w2c, scale, gt_img, steps, pipe, bg):
    """Uni3R's official test-time target-pose optimisation (uni3r/loss.py TestLoss +
    loss_of_one_batch): 9-dof delta (3 trans + 6d rot), Adam lr 5e-3, RGB L1, `steps`
    iterations, delta right-multiplied onto w2c. Uses the target GT image -> must be
    reported separately from the strict SIU3R protocol."""
    import copy
    from uni3r.loss import rotation_6d_to_matrix
    # RGB-only copy: the loss only uses RGB, so do not rasterise the 64-d feature field
    # (same geometry/colours; equivalent to TestLoss fast_pose_alignment, ~16x fewer channels)
    gaussians = copy.copy(gaussians)
    gaussians._semantic_feature = torch.zeros_like(gaussians._semantic_feature[..., :1])
    delta = torch.nn.Parameter(torch.zeros(1, 9, device=w2c.device))
    opt = torch.optim.Adam([delta], lr=5e-3)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=w2c.device)

    def cur():
        T = torch.eye(4, device=w2c.device)
        T[:3, :3] = rotation_6d_to_matrix(delta[:, 3:] + ident)[0]
        T[:3, 3] = delta[0, :3]
        return w2c @ T
    with torch.enable_grad():
        for _ in range(steps - 1):  # same count as TestLoss: steps-1 updates, then evaluate
            img = render(cam, gaussians, pipe, bg, intrinsics=K, extrinsics=cur(), scale=scale)['render']
            loss = (img[0].permute(2, 0, 1) - gt_img).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return cur().detach()


def load_siu3r(siu3r_repo):
    sys.path.insert(0, str(siu3r_repo))
    from src.data.components.scannet_dataset import ScanNetDataset
    from src.visualizer import Visualizer
    from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME
    assert [PANOPTIC_SEMANTIC2NAME[i] for i in range(1, 21)] == SIU3R_CLASSES
    return ScanNetDataset, Visualizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--siu3r_repo', required=True)
    ap.add_argument('--root', required=True, help='dir with val/<scan> and val_pair.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', default=str(REPO / 'checkpoints/output/2views.pth'))
    ap.add_argument('--shard_count', type=int, default=1)
    ap.add_argument('--shard_index', type=int, default=0)
    ap.add_argument('--max_pairs', type=int, default=0)
    ap.add_argument('--pose_align_steps', type=int, default=0,
                    help='0 = strict SIU3R protocol; 100 = Uni3R official test-time alignment')
    args = ap.parse_args()

    ScanNetDataset, Visualizer = load_siu3r(args.siu3r_repo)
    ds = ScanNetDataset(root=args.root, train=False, seg_task='panoptic',
                        num_extra_context_views=0, num_extra_target_views=4,
                        val_pair_json='val_pair.json')
    available = set(ds.scan_names)
    todo = [i for i, p in enumerate(ds.val_pairs) if p['scan'] in available]
    todo = todo[args.shard_index::args.shard_count]
    if args.max_pairs:
        todo = todo[:args.max_pairs]
    print(f'pairs with data: {sum(p["scan"] in available for p in ds.val_pairs)}/'
          f'{len(ds.val_pairs)}; this shard: {len(todo)}', flush=True)

    from types import SimpleNamespace
    vis = Visualizer(SimpleNamespace(log_colored_depth=False, log_rendered_video=False,
                                     log_gaussian_ply=False, save_sh_dc_only=True,
                                     dataset_name='scannet', overlay_mask_alpha=0.5,
                                     write_to=args.out))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    model = Uni3R.from_pretrained(args.checkpoint, device=device, num_views_value=2).eval()
    pipe, bg = DummyPipeline(), torch.zeros(3, device=device)
    skipped = []

    for n, idx in enumerate(todo):
        pair = ds.val_pairs[idx]
        tag = f"{pair['scan']}_context{'_'.join(map(str, pair['context_ids']))}"
        if (out / tag / 'target_seg_gt').exists():
            continue  # resume
        d = ds[idx]
        if d['scene_names'] != pair['scan'] or list(d['context_views_id']) != list(pair['context_ids']):
            # SIU3R's dataset silently moves to the next pair on invalid labels; its own
            # visualizer then never writes this pair. Mirror that: skip it.
            skipped.append(tag)
            continue
        H, W = 256, 256
        ctx_img = torch.tensor(np.array(d['context_views_images'])).float() / 255.0  # (2,3,H,W)
        tgt_img = torch.tensor(np.array(d['target_views_images'])).float() / 255.0  # (N,3,H,W)
        ctx_K = torch.tensor(np.array(d['context_views_intrinsics']), dtype=torch.float32)
        tgt_K = torch.tensor(np.array(d['target_views_intrinsics']), dtype=torch.float32)
        ctx_K[:, 0] *= W; ctx_K[:, 1] *= H; tgt_K[:, 0] *= W; tgt_K[:, 1] *= H  # back to pixels
        ctx_E = np.array(d['context_views_extrinsics'], dtype=np.float32)
        tgt_E = np.array(d['target_views_extrinsics'], dtype=np.float32)
        baseline = float(np.linalg.norm(ctx_E[1, :3, 3]))

        views = [{'img': (ctx_img[i:i + 1] * 2 - 1).to(device),
                  'camera_intrinsics': ctx_K[i:i + 1].to(device)} for i in range(2)]
        with torch.no_grad():
            preds = model(views)
            g = merge_and_split_predictions(*preds)[0]
            gaussians = GaussianModel.from_predictions(g, sh_degree=3)
            rgbs, depths, sems = [], [], []
            for j in range(len(d['target_views_id'])):
                c2w = torch.tensor(tgt_E[j]).to(device)
                c2w[:3, 3] /= baseline
                w2c = torch.eye(4, device=device)
                w2c[:3, :3] = c2w[:3, :3].T
                w2c[:3, 3:4] = -c2w[:3, :3].T @ c2w[:3, 3:4]
                K = tgt_K[j].to(device)
                cam = get_scaled_camera(None, c2w, K, 1, torch.tensor([H, W]))
                if args.pose_align_steps:
                    w2c = align_pose(gaussians, cam, K, w2c, baseline, tgt_img[j].to(device),
                                     args.pose_align_steps, pipe, bg)
                r = render(cam, gaussians, pipe, bg, intrinsics=K, extrinsics=w2c, scale=baseline)
                rgbs.append(r['render'][0].permute(2, 0, 1).clamp(0, 1))
                # render() rescales the scene by 10*baseline (near-plane trick): depth/10 = metres
                depths.append(r['depth'][0, ..., 0] / 10.0)
                feat = model.feature_expansion(r['feature_map'].permute(0, 3, 1, 2))
                logits = model.lseg_feature_extractor.decode_feature(feat, PROMPTS)
                assert logits.shape[-2:] == (H, W) and logits.shape[1] == 20, logits.shape
                sems.append(logits.argmax(1)[0] + 1)
        rgbs = torch.stack(rgbs)[None].cpu()
        depths = torch.stack(depths)[None].cpu()
        sem_t = torch.stack(sems)[None].cpu().long()
        ins_t = torch.zeros_like(sem_t)
        ctx_pos = [p for p, t in enumerate(d['target_views_id']) if t in d['context_views_id']]
        assert len(ctx_pos) == 2
        gt_depths = torch.tensor(np.array(d['target_views_depths']), dtype=torch.float32)[None]

        scan, cids, tids = [d['scene_names']], [d['context_views_id']], [d['target_views_id']]
        vis.visualize_recon_image(rgbs, depths, tgt_img[None], gt_depths, str(out), scan, cids, tids)
        vis.save_seg_ids('context', sem_t[:, ctx_pos], ins_t[:, ctx_pos], str(out), scan, cids,
                         seg_infos=[[]])
        vis.save_seg_ids('target', sem_t, ins_t, str(out), scan, cids, tids, seg_infos=[[]])
        vis.save_gt_seg_masks('context', [d['context_mask_labels']], [d['context_class_labels']],
                              str(out), scan, cids)
        vis.save_gt_seg_masks('target', [d['target_mask_labels']], [d['target_class_labels']],
                              str(out), scan, cids, tids)
        if n % 20 == 0:
            print(f'[{args.shard_index}] {n}/{len(todo)} {tag}', flush=True)
    (out / f'skipped_shard{args.shard_index}.txt').write_text('\n'.join(skipped))
    print(f'DONE shard {args.shard_index}: {len(todo)} pairs, skipped {len(skipped)}', flush=True)


if __name__ == '__main__':
    main()
