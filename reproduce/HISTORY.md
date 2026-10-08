# Active goal: current handoff (2026-10-05 18:25 UTC)

Goal remains active: run Uni3R and full ScanNet 2/8/16-view evaluations on local L20.
Network workaround verified: local SOCKS `socks5h://127.0.0.1:10808` reaches Hugging
Face and Google Drive. HTTP proxy port 7890 failed TLS; use 10808. Local setup and
runner export this proxy. Checkpoint downloader resolves through the proxy, downloads
CDN parts directly except slow `part_aa` via proxy, resumes each attempt, and verifies SHA256.

Completed:
- ScanNet archive hash verified and extracted; root in EFS `data_root.txt`.
- Dataset checks: 2 views 320 samples/40 scenes; 8/16 views 49 scenes each.
- VGGT fixed-code and LSeg fixed-source present; simple_knn+CLIP installed.
- L20 nearest-neighbor and gsplat render/backward checks passed; gsplat extension cached
  in repository `reproduce/build_cache` (ignored). Peak test tensor memory 8.25 MiB.
- LSeg checkpoint `checkpoints/demo_e200.ckpt`, VGGT `cache/torch/hub/checkpoints/model.pt`,
  CLIP and LPIPS VGG16 weights downloaded. CLIP user-cache symlink points to EFS.
- Lightning 2.5.1 installed for LSeg checkpoint serialization. LSeg smoke is running,
  currently constructing the model (which may download timm's initial ViT weights).

Running:
- `reproduce/fetch_checkpoints.py`, log EFS `logs/resolve_weights.log`: prioritizes 2/8/16.
  2-view parts ab and ac are verified; aa is incomplete. Other partial downloads retained.
- `/tmp/uni3r_lseg_smoke.py`, log EFS `logs/lseg_smoke.log`, GPU 0.

Next: finish/check LSeg smoke, assemble each view checkpoint after all its parts verify,
then run `UNI3R_GPU=0 UNI3R_VIEWS=2 UNI3R_MAX_BATCHES=1 bash reproduce/run_local.sh`
after checking current GPU capacity. Fix actual runtime issues, then omit MAX_BATCHES
for full metrics. No full-model inference or paper metrics have completed yet.

Checkpoint metadata shows pretrained distiller is absent by design; loader allows
missing distiller parameters because it initializes them from VGGT separately. Other
missing prediction parameters are rejected. mmap=True is used for checkpoint reads.

---

# Active goal update (2026-10-05 UTC)

Goal: complete model inference and full ScanNet 2/8/16-view evaluations on existing
local L20 GPUs, with no cloud jobs. Both previously submitted cloud jobs are cancelled.

- Test data downloaded, SHA256 verified, extracted at EFS `data/data/scannet_test`.
- CPU dataset loading passed: 2 views = 320 samples / 40 scenes; 8 and 16 views =
  49 samples / 49 scenes each, matching the unmodified upstream loader.
- VGGT code fetched at gitlink revision and verified against Git blob hashes;
  LSeg archive passed gzip integrity and was extracted at the fixed gitlink revision.
- Python imports passed. simple-knn and CLIP built/installed. PyTorch3D is now a lazy
  training-only dependency, so evaluation does not require it.
- L20 nearest-neighbor distances matched a torch.cdist reference. gsplat rendering
  JIT build is in progress after fixing ninja PATH. Log: `logs/backend_check.log`.
- No model inference or final metrics yet. Official fixed-view checkpoint downloads
  remain blocked by repeated mirror TCP connection timeouts; LSeg Google Drive is
  also unreachable. Download resolver is retrying; `logs/resolve_weights.log`.
- Local runner: `reproduce/run_local.sh`; setup: `reproduce/setup_local.sh`.
- `test.py --max_batches 1` produces explicitly marked smoke-run metrics, never
  full-evaluation metrics. Checkpoint loading now rejects missing non-LSeg weights.

---

# Download update (2026-10-05 17:39 UTC)

ScanNet test archive successfully downloaded to EFS `Uni3R/data/scannet_test.tar`
and its SHA256 verified against the upstream LFS hash. Hugging Face direct TCP
connection still timed out, but hf-mirror returned a fresh CDN redirect and the
CDN download succeeded at approximately 17 MB/s. Asset preparation is now extracting
the test archive and downloading published checkpoint parts; logs: `logs/assets.log`.
The official 2/8/16-view checkpoints are split files and the downloader now handles
ordered concatenation with size and individual SHA256 checks. No evaluation scores yet.

---

# Execution update (2026-10-05 UTC)

User clarified that L20 means the existing development machine. Both cloud jobs
`t-20261006010211-sns8j` (A800) and `t-20261006011814-k9kcj` (L20) were cancelled.
No new cloud allocation is needed. Use `reproduce/run_local.sh` for local evaluation.
All eight local L20s currently have active compute processes; evaluate only after
assets are ready and GPU capacity has been checked. No evaluation scores exist yet.
VGGT source archive download failed with curl HTTP/2 error 92; source downloader now
uses HTTP/1.1 and retries transport errors. Dataset/model downloads remain incomplete.

---

Historical preparation record follows; cloud queue details below are superseded.

# Uni3R ScanNet reproduction

## Current status (2026-10-05 UTC)

At the user’s request, the original A800 task `t-20261006010211-sns8j`
was cancelled successfully and replaced with the L20 task below.
The L20 queue currently has all 8 GPUs allocated; the new job is queued.

Cloud task `t-20261006011814-k9kcj` was submitted successfully and last observed in `Queue`.
Resources: one worker, **3 NVIDIA L20 48GB**, 24 CPUs, 240 GiB RAM.
Queue: `q-20260718133137-lhvrm`. Deadline: 24 hours. No automatic retries.

**No evaluation has completed; no reproduced scores are available yet.**
Hugging Face and hf-mirror connections repeatedly timed out from the development host.
The ScanNet archive and model checkpoints have not been downloaded successfully.
The queued task will retry preparation from the allocated cloud worker before evaluating.

Storage: `/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R`.
The Python 3.10 venv inherits torch 2.7.1+cu128 from the existing fastw environment;
additional dependencies are isolated in the venv. CUDA 12.9 was copied to EFS for building
L20 CUDA extensions (compute capability 8.9). Python package installation completed; CUDA extension builds and
model inference still require verification. Source preparation uses GitHub codeload because
the git endpoint failed. The two Uni3R submodules use their exact gitlink revisions.

## Run and inspect

```bash
export PATH="$HOME/.volc/bin:$PATH"
volc ml_task get --id t-20261006011814-k9kcj --output json
volc ml_task logs --task t-20261006011814-k9kcj --instance worker_0
```

The entrypoint is `reproduce/run_cloud.sh`; config is `reproduce/task.yaml`.
Logs and results are saved under EFS `runs/<UTC timestamp>/`:
`driver.log`, `exit_code`, and `{2,8,16}views/{eval.log,metrics.json}`.
Package versions and the code diff are recorded in each run.
Local preparation logs are EFS `logs/install.log` and `logs/sources.log`.
vePFS `.setup.lock` serializes local/cloud setup; EFS does not support flock.

## Evaluation protocol and reference values

Source: https://arxiv.org/html/2508.03643, Tables 1 and 2 (retrieved 2026-10-05).
These are **paper reference values**, not measured results.

| Views | PSNR | SSIM | LPIPS | Target mIoU | Target accuracy |
|---|---:|---:|---:|---:|---:|
| 2 | 25.53 | 0.8727 | 0.1380 | 0.5584 | 0.8268 |
| 8 | 24.71 | 0.851 | — | 0.554 | 0.807 |
| 16 | 23.32 | 0.816 | — | 0.558 | 0.835 |

Use released fixed-view checkpoints, 256x256 resolution, seed=777, batch size=1,
100 pose alignment steps, AMP disabled. Each GPU evaluates an entire view setting;
no distributed sampler or metric averaging across the three runs is used.
The dataset's original `scene0696_02` exclusion is preserved.
The published test archive is about 1.07 GB, with expected SHA256
`d327b360c0743a9f7cc9d3fe895baaa3aa9ffbbbbae960b58ee9fe84e5d83a45`.
The full ScanNet training dataset is not needed for these evaluations.

## Changes

- Fix `Uni3R.from_pretrained(..., device=device)`; the old positional argument was
  passed into the `use_pretrained_lseg` flag instead of the device parameter.
- Explicit `weights_only=False` for trusted upstream checkpoints under torch >=2.6.
- `UNI3R_LSEG_CHECKPOINT` overrides training-machine paths stored in checkpoints.
- Persist metrics as JSON without changing the upstream metric computation.
- Add resumable assets download, checksum checking, source preparation, and cloud runner.

Syntax checks passed. End-to-end import, CUDA compilation, inference, and paper-score
comparison are not yet verified. The modern PyTorch/gsplat versions differ from the
README's original environment; any score discrepancy must be reported and investigated.

The original NVlabs/LSM data guide was checked and points to the same Hugging Face
archive; no independent test-set mirror has been verified. Raw official ScanNet
download requires existing access authorization and exact reproduction of the split.


---

# Uni3R reproduction status (2026-10-05 19:10 UTC)

Active goal: complete full 2/8/16-view evaluation on existing development L20 GPUs.
No cloud GPU allocation is needed. Earlier cloud jobs were cancelled.

## Verified

- ScanNet test archive downloaded, SHA256 verified, extracted on EFS.
- Original dataset loader: 2 views = 320 samples / 40 scenes; 8 and 16 views = 49 samples / 49 scenes.
- All official checkpoint split files for 2/8/16 views downloaded and SHA256 verified.
- All three checkpoints assembled; combined hashes and ZIP structure verified.
- CUDA nearest-neighbor and Gaussian render/backward checks passed on L20.
- Official LSeg checkpoint strict loading, feature extraction and decoding passed.
- Complete 2-view one-sample inference, rendering, 100-step alignment and metrics passed.
  Baseline: `runs/local-20261005T184511Z/2views/metrics.json` on EFS.
  PSNR 24.6494, SSIM 0.804872, LPIPS 0.123168. These are smoke results, not full scores.

## In progress

An optional `TestLoss(fast_pose_alignment=True)` skips semantic computation during
intermediate alignment iterations, where only RGB L1 contributes to the loss.
Rendering, RGB objective, 100 alignment steps and final metrics are retained.
Same-sample validation passed: all RGB, semantic and depth metrics are exactly equal.
Sample evaluation time decreased from about 151.5 to 6.57 seconds.
Evidence: alignment_validation.json. Full 2-view evaluation completed on GPU 6 with exit code 0 (320/320):
PSNR 25.370596, SSIM 0.867149, LPIPS 0.141197, mIoU 0.546484, accuracy 0.819008.
Full run: `runs/local-20261005T190036Z/2views/metrics.json`.
See RESULTS_2views.md for measured-minus-paper differences;
8-view smoke passed (peak CUDA allocation ~16 GB), and full 8-view evaluation is
running on GPU 0. The 16-view smoke passed (peak CUDA allocation ~18.4 GB); full 16-view evaluation
is now running on GPU 5. First-sample time was ~97 seconds; full run ETA needs
more samples because shared GPU contention and warmup affect timings.
Full 8/16-view dataset scores are still outstanding.

## Reproduce

Repository: `/file_system/vepfs/intern/haozhe.jia/projects/FastWAM/3DGS/Uni3R`
Storage: `/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R`

```bash
# Select an L20 with sufficient free memory using nvidia-smi first.
UNI3R_GPU=5 UNI3R_VIEWS="2 8 16" UNI3R_FAST_ALIGNMENT=True bash reproduce/run_local.sh
# For a smoke test only, add UNI3R_MAX_BATCHES=1.
# UNI3R_FAST_ALIGNMENT=False restores the original intermediate computations.
```

The runner uses 256x256, seed 777, batch size 1, AMP disabled, 100 alignment steps.
It saves logs, metrics, package versions, source diff and the LSeg vendor patch on EFS.
The working download proxy is `http://127.0.0.1:37890`; direct Hugging Face times out.
Resolved signed CDN links allow direct transfers. Downloads now completed.

## Interpretation caveats

- Upstream MeanIoU returns -1 for classes absent throughout evaluation; baseline
  smoke mIoU is therefore negative. The installed implementation averages each class
  over updates where its union is nonzero. Full evaluation retains this aggregation.
- Paper section 4.1 explicitly states 40 unseen ScanNet scenes. The released
  multi-view loader uses 49 scenes, whereas 2-view uses the selected 40. These
  runs preserve the released loader; 8/16-view paper comparisons are therefore
  indicative rather than an exact match of the stated paper protocol.
- Reported depth metrics are target-view; paper table 1 depth is source-view.
- Environment uses torch 2.7.1 / gsplat 1.5.3; numerical differences require checking.

Paper references (not reproduced scores): https://arxiv.org/html/2508.03643

| Views | PSNR | SSIM | LPIPS | Target mIoU | Target accuracy |
|---|---:|---:|---:|---:|---:|
|2|25.53|0.8727|0.1380|0.5584|0.8268|
|8|24.71|0.851|—|0.554|0.807|
|16|23.32|0.816|—|0.558|0.835|

Earlier records are in HISTORY.md and are superseded by this status.

## Additional validation (19:16 UTC)

An RGB-only intermediate-render experiment on the same 16-view sample reduced
latency from 97.30 to 37.80 seconds but changed PSNR by -0.000198 dB and other
metrics slightly. It was not adopted; all experiment-only production changes
were reverted. Existing full runs continue unchanged. Numerical evidence is
in rgb_alignment_experiment.json and the experiment run stores its source patch.
Both full 8/16-view process handles were re-polled and remain live.

## Full 8-view result (19:22 UTC)

Full 8-view evaluation exited 0 with all 49/49 scenes processed.
PSNR 24.1222, SSIM 0.8298, LPIPS 0.250930, mIoU 0.5213, accuracy 0.8067.
Original metrics: `runs/local-20261005T190318Z/8views/metrics.json` on EFS.
See RESULTS_8views.md for exact measured-minus-paper values.
Only full 16-view evaluation remains (session 15947, PID 1377538, GPU 5).
`make_report.py` creates final repo/EFS reports only after all full runs exit 0.

## 16-view progress (19:31:56 UTC)

Latest logged batch index 20/49 (21 scenes completed). Average recent time
67.08 s/scene; log ETA 33m07s. Peak CUDA allocation 23026 MiB.
Process 1377538 and session 15947 are live; no runtime errors observed.
