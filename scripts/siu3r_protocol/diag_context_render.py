"""Diagnose low PSNR when re-rendering a context view (identity pose) on SIU3R data.

Compares, for a few val pairs, the PSNR of rendering context view 1 under:
  A) SIU3R GT intrinsics, identity pose (what export_uni3r.py does)
  B) Uni3R's own predicted intrinsics/extrinsics for that view
  C) A + Uni3R's 100-step target-pose optimisation (its official test protocol)
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
from export_uni3r import (init_all_submodules, Uni3R, merge_and_split_predictions,  # noqa
                          get_scaled_camera, DummyPipeline, render, GaussianModel, load_siu3r)
from uni3r.loss import rotation_6d_to_matrix  # noqa


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def main():
    ScanNetDataset, _ = load_siu3r('/mnt/workspace/Jensen/project/3DGS/external/SIU3R')
    ds = ScanNetDataset(root='/mnt/workspace/Jensen/project/3DGS/data/siu3r_protocol_root',
                        train=False, seg_task='panoptic', num_extra_context_views=0,
                        num_extra_target_views=4)
    dev = torch.device('cuda')
    model = Uni3R.from_pretrained(str(REPO / 'checkpoints/output/2views.pth'), device=dev,
                                  num_views_value=2).eval()
    pipe, bg = DummyPipeline(), torch.zeros(3, device=dev)
    avail = set(ds.scan_names)
    idxs = [i for i, p in enumerate(ds.val_pairs) if p['scan'] in avail][:8]
    for idx in idxs:
        d = ds[idx]
        H = W = 256
        ctx = torch.tensor(np.array(d['context_views_images'])).float() / 255
        K = torch.tensor(np.array(d['context_views_intrinsics']), dtype=torch.float32)
        K[:, 0] *= W; K[:, 1] *= H
        E = np.array(d['context_views_extrinsics'], dtype=np.float32)
        base = float(np.linalg.norm(E[1, :3, 3]))
        views = [{'img': (ctx[i:i + 1] * 2 - 1).to(dev), 'camera_intrinsics': K[i:i + 1].to(dev)}
                 for i in range(2)]
        with torch.no_grad():
            preds = model(views)
        g = GaussianModel.from_predictions(merge_and_split_predictions(*preds)[0], sh_degree=3)
        gt0, gt1 = ctx[0].to(dev), ctx[1].to(dev)

        def rend(c2w, Kt, scale=base):
            w2c = torch.linalg.inv(c2w)
            cam = get_scaled_camera(None, c2w, Kt, 1, torch.tensor([H, W]))
            return render(cam, g, pipe, bg, intrinsics=Kt, extrinsics=w2c, scale=scale)['render'][0].permute(2, 0, 1)

        I = torch.eye(4, device=dev)
        c1 = torch.tensor(E[1]).to(dev); c1[:3, 3] /= base
        with torch.no_grad():
            a0 = psnr(rend(I, K[0].to(dev)).clamp(0, 1), gt0)
            a1 = psnr(rend(c1, K[1].to(dev)).clamp(0, 1), gt1)
            pk = preds[0]['intr'][0].float()
            pe = preds[1]['extr'][0].float()  # predicted w2c? report both conventions
            b0 = psnr(rend(I, pk).clamp(0, 1), gt0)
            b1 = max(psnr(rend(torch.linalg.inv(pe), pk).clamp(0, 1), gt1),
                     psnr(rend(pe, pk).clamp(0, 1), gt1))
        # C) optimise a pose delta for view 1 (same parametrisation as TestLoss)
        delta = torch.zeros(1, 9, device=dev, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=5e-3)
        ident6 = torch.tensor([1., 0, 0, 0, 1, 0], device=dev)
        for _ in range(100):
            R = rotation_6d_to_matrix(delta[:, 3:] + ident6)[0]
            T = torch.eye(4, device=dev); T[:3, :3] = R; T[:3, 3] = delta[0, :3]
            w2c = torch.linalg.inv(c1) @ T
            cam = get_scaled_camera(None, c1, K[1].to(dev), 1, torch.tensor([H, W]))
            img = render(cam, g, pipe, bg, intrinsics=K[1].to(dev), extrinsics=w2c, scale=base)['render'][0].permute(2, 0, 1)
            loss = (img - gt1).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        c_1 = psnr(img.detach().clamp(0, 1), gt1)
        print(f"{d['scene_names']} {d['context_views_id']} base={base:.3f}m | "
              f"A ctx0={a0:.2f} ctx1={a1:.2f} | B(pred K/E) ctx0={b0:.2f} ctx1={b1:.2f} | "
              f"C ctx1+align={c_1:.2f}", flush=True)


if __name__ == '__main__':
    main()
