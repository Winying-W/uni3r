#!/usr/bin/env python3
"""A M0: real diagnostic fixture + original-forward/RGB/student regression.

Uses the original official TestDataset preprocessing. This is NOT a training
dataset exporter: formal train/val inputs and instance targets remain C-owned.
Run after `source scripts/env.sh`; select a GPU with CUDA_VISIBLE_DEVICES.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from uni3r.utils.path_manager import init_all_submodules
init_all_submodules()
from uni3r.instance.frozen_base import FrozenUni3R, map_tensors, tensor_items
from uni3r.instance.contracts import TOKEN_LAYERS, validate_frozen_scene_batch

EXPECTED_SHA = "660aed13d6a8bccd078be35210f803d0f6d3b312f63dd468bbccc1c7f3aacf24"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_provenance(out):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    manifest = {}
    # Includes current uncommitted A/B code and vendored model/CUDA sources.
    for folder in ("uni3r", "instance_B", "scripts", "submodules"):
        for p in sorted((ROOT / folder).rglob("*")):
            if (p.is_file() and p.suffix in {".py", ".cpp", ".cu", ".cuh", ".h", ".sh"}
                    and not set(p.relative_to(ROOT).parts) & {".git", "__pycache__", "build", "results"}):
                manifest[str(p.relative_to(ROOT))] = sha256(p)
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    (out / "source_manifest.json").write_text(serialized)
    return dict(uni3r_git_commit=git("rev-parse", "HEAD"),
                uni3r_worktree_dirty=bool(git("status", "--porcelain")),
                uni3r_source_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                submodule_status=git("submodule", "status"))


def load_pair(data_root, index):
    from uni3r.datasets.testdata import TestDataset
    from dust3r.datasets.utils import cropping

    class TracedDataset(TestDataset):
        def _crop_resize_if_necessary(self, image, depth, K, resolution, **kwargs):
            events = []
            original_size = list(image.shape[:2])
            crop_fn, resize_fn = cropping.crop_image_depthmap, cropping.rescale_image_depthmap

            def crop(*args, **kw):
                result = crop_fn(*args, **kw)
                events.append(dict(operation="crop", input_wh=list(args[0].size),
                                   bbox_xyxy=[int(x) for x in args[3]], output_wh=list(result[0].size)))
                return result

            def resize(*args, **kw):
                result = resize_fn(*args, **kw)
                events.append(dict(operation="resize", input_wh=list(args[0].size),
                                   output_wh=list(result[0].size),
                                   rgb_interpolation="PIL Lanczos if downscale else bicubic",
                                   label_interpolation="OpenCV nearest (original loader)"))
                return result

            with patch.object(cropping, "crop_image_depthmap", crop), patch.object(cropping, "rescale_image_depthmap", resize):
                result = super()._crop_resize_if_necessary(image, depth, K, resolution, **kwargs)
            path = Path(kwargs["info"])
            self.trace[str(path)] = dict(original_size=original_size, crop_xyxy=events[0]["bbox_xyxy"],
                resize_hw=[256, 256], operations=events,
                description="Original TestDataset: principal-point crop, aspect-preserving resize, final centered crop. crop_xyxy records FIRST crop; operations records full ordered transform, including final crop.",
                source_rgb=str(path.resolve()), source_rgb_sha256=sha256(path),
                source_camera_npz_sha256=sha256(path.with_suffix(".npz")),
                source_intrinsics=K.tolist(), output_intrinsics=result[2].tolist())
            return result

    dataset = TracedDataset(ROOT=str(data_root), split="test", resolution=(256, 256), seed=777, num_views=2)
    dataset.trace = {}
    views = dataset[index][:2]
    scenes = [v["label"] for v in views]
    if len(set(scenes)) != 1:
        raise ValueError("Pair must be from one scene")
    frames = [v["instance"] for v in views]
    images = torch.stack([v["img"] for v in views])[None]
    K = torch.from_numpy(np.stack([v["camera_intrinsics"] for v in views]))[None]
    meta = dict(scene_id=[scenes[0]], frame_ids=[frames], crop_resize=[[
        dataset.trace[str(data_root / s / "images" / f)] for s, f in zip(scenes, frames)]],
        purpose="M0_DIAGNOSTIC_ONLY_NOT_FORMAL_TRAIN_OR_VAL", dataset="official Uni3R scannet_test",
        dataset_sample_index=index, dataset_seed=777,
        encoder_input_fields=["RGB", "cropped pixel intrinsics"],
        target_status="No InstanceTargets exported; test semantic maps are not instance IDs.")
    return images, K, meta


def compare(a, b, name, *, atol=1e-6, rtol=1e-5):
    aa, bb = dict(tensor_items(a)), dict(tensor_items(b))
    if aa.keys() != bb.keys():
        raise AssertionError(f"{name}: tensor keys differ")
    result = {}
    for key in aa:
        x, y = aa[key].detach().cpu(), bb[key].detach().cpu()
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol, msg=lambda m: f"{name}.{key}: {m}")
        result[key] = {"shape": list(x.shape), "max_abs": float((x-y).abs().max()), "exact": torch.equal(x, y)}
    return result


@torch.no_grad()
def render_original(views):
    from uni3r.utils.gaussian_model import GaussianModel
    from uni3r.utils.cuda_splatting import render
    keys = ("means", "rotations", "scales", "opacities", "sh_coeffs", "gs_feats")
    flat = {k: torch.cat([p[k][0].flatten(0, 1) for p in views]) for k in keys}
    model = GaussianModel.from_predictions(flat, sh_degree=3)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=True, debug=False)
    results = []
    for v in views:
        extr, intr = v["extr"][0], v["intr"][0]
        camera = SimpleNamespace(image_width=256, image_height=256, camera_center=torch.linalg.inv(extr)[:3, 3])
        results.append(render(camera, model, pipe, torch.zeros(3, device=extr.device),
                              intrinsics=intr, extrinsics=extr, scale=None)["render"])
    return torch.cat(results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/output/2views.pth")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/scannet_test")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    p.add_argument("--check-student", action="store_true")
    p.add_argument("--dpt-init-dir", type=Path, default=ROOT / "instance_B/results/B-01/20261008_cpu_01")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    torch.manual_seed(777)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print("Hashing official checkpoint and current source...", flush=True)
    checksum = sha256(args.checkpoint)
    if checksum != EXPECTED_SHA:
        raise ValueError("Checkpoint is not the verified official phase-one 2views.pth")
    provenance = source_provenance(out)
    provenance["base_checkpoint"] = dict(name="2views.pth", sha256=checksum, path=str(args.checkpoint.resolve()))
    images, K, sample_meta = load_pair(args.data_root, args.sample_index)
    print("Pair:", sample_meta["scene_id"], sample_meta["frame_ids"], flush=True)
    images, K = images.cuda(), K.cuda()
    from uni3r.uni3r import Uni3R
    base = Uni3R.from_pretrained(str(args.checkpoint), device="cuda", num_views_value=2).eval()
    base.requires_grad_(False)
    contexts = [{"img": images[:, i], "camera_intrinsics": K[:, i]} for i in range(2)]
    reference = {}

    def reference_tokens(_m, _i, output):
        t = output["tokens"]
        reference["tokens"] = {l: t["aggregated_tokens_list"][l][:, :, t["patch_start_idx"]:].detach().cpu().clone() for l in TOKEN_LAYERS}

    def reference_dpt(_m, _i, output):
        reference["dpt"] = output.detach().cpu().clone()

    handles = [base.vggt.register_forward_hook(reference_tokens), base.dpt_head.register_forward_hook(reference_dpt)]
    print("Running unwrapped original forward...", flush=True)
    try:
        with torch.no_grad():
            original = base(contexts)
            rgb = render_original(original).cpu()
    finally:
        for handle in handles:
            handle.remove()
    original_cpu = map_tensors(original, lambda t: t.detach().cpu())
    torch.save({"views": original_cpu, "rgb": rgb, "metadata": provenance}, out / "reference.pt")
    del original
    print("Running frozen wrapper...", flush=True)
    wrapper = FrozenUni3R(base, provenance).train()
    assert all(not m.training for m in base.modules())
    online = wrapper(images, K, sample_meta)
    report = dict(status="RUNNING", purpose=sample_meta["purpose"], scene_id=sample_meta["scene_id"],
                  frame_ids=sample_meta["frame_ids"], provenance=provenance,
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(), seed=777, autocast=False, tf32=False,
                  tolerances=dict(atol=1e-6, rtol=1e-5))
    report["original_vs_wrapper"] = compare(original_cpu, online["gaussians"]["views"], "original_vs_wrapper")
    report["reference_tokens_vs_wrapper"] = compare(reference["tokens"], online["patch_tokens"], "tokens")
    report["rgb_regression"] = compare(rgb, render_original(online["gaussians"]["views"]), "RGB")
    report["render_protocol"] = "Original cuda_splatting.render, predicted source cameras, scale=None, GaussianModel default active_sh_degree=0, raw rotations unchanged. Regression only; no GT camera or novel-view quality claim."
    cpu = map_tensors(online, lambda t: t.detach().cpu())
    validate_frozen_scene_batch(cpu)
    torch.save(cpu, out / "fixture.pt")
    (out / "metadata.json").write_text(json.dumps(cpu["metadata"], indent=2) + "\n")
    for i in range(2):
        Image.fromarray(((images[0, i].permute(1, 2, 0).cpu().numpy()+1)*127.5).clip(0, 255).round().astype(np.uint8)).save(out / f"input_{i}.png")
        Image.fromarray((rgb[i].numpy().clip(0, 1)*255).round().astype(np.uint8)).save(out / f"reference_rgb_{i}.png")
    loaded = torch.load(out / "fixture.pt", map_location="cuda", weights_only=True)
    validate_frozen_scene_batch(loaded)
    report["cache_roundtrip"] = compare(online, loaded, "cache", atol=0, rtol=0)
    if args.check_student:
        print("Checking B default student with real online and cached inputs...", flush=True)
        from instance_B.student import InstanceStudent, StudentConfig, load_dpt_initialization
        init = load_dpt_initialization(args.dpt_init_dir / "dpt_init_from_2views.pt", args.dpt_init_dir / "dpt_init_receipt.json")
        student = InstanceStudent(StudentConfig(), dpt_init=init).cuda().eval()
        with torch.no_grad():
            dense = student.instance_dpt(online["patch_tokens"], online["images"])
            report["copied_dpt_vs_original"] = compare(reference["dpt"], dense, "DPT")
            del dense
            pred_online = student(online)
            pred_cached = student(loaded)
        report["student_cache_vs_online"] = compare(pred_online, pred_cached, "student", atol=0, rtol=0)
        del pred_online, pred_cached, loaded
        # A gradient-isolation diagnostic, NOT instance training or a quality metric.
        versions = {n: (p.data_ptr(), p._version) for n, p in base.named_parameters()}
        base_ids = {id(p) for p in base.parameters()}
        assert not any(id(p) in base_ids for p in student.parameters())
        before = student.object_queries.detach().clone()
        student.train()
        optimizer = torch.optim.SGD(student.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        prediction = student(online)
        loss = prediction["objectness_logits"].square().mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        assert float(grad_norm) > 0
        optimizer.step()
        assert not torch.equal(before, student.object_queries)
        assert all(p.grad is None and not p.requires_grad for p in base.parameters())
        assert versions == {n: (p.data_ptr(), p._version) for n, p in base.named_parameters()}
        report["gradient_isolation"] = dict(base_gradients=0, base_parameter_versions_unchanged=True,
            student_gradient_norm=float(grad_norm), student_queries_changed=True,
            diagnostic_loss="mean(objectness_logits**2), no labels; step discarded, NOT instance training")
        report["student_status"] = "Default B student with verified copied DPT and random other heads; no trained instance checkpoint."
    report["all_base_parameters_frozen"] = all(not p.requires_grad and p.grad is None for p in base.parameters())
    report["base_train_protected"] = all(not m.training for m in base.modules())
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/1024**3
    report["artifact_sha256"] = {f: sha256(out / f) for f in ("fixture.pt", "reference.pt", "metadata.json", "source_manifest.json")}
    report["status"] = "A_M0_REAL_FIXTURE_VERIFIED"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "scene_id", "frame_ids", "peak_allocated_gib")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
