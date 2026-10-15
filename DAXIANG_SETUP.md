# Daxiang deployment

All project storage belongs on `/file_system/efs/checkpoint/intern/ying.wu`.
The code checkout is `Uni3R/`; weights, datasets, caches, temporary files and
environments live alongside it on the same EFS mount.

```bash
cd /file_system/efs/checkpoint/intern/ying.wu
git clone --recurse-submodules https://github.com/Winying-W/uni3r.git Uni3R
cd Uni3R
bash scripts/setup_daxiang_efs.sh
source scripts/env_daxiang.sh
python3 scripts/download_reproduction.py --views 2 --weights-only
```

The download script obtains official Uni3R 2-view, VGGT, LSeg, CLIP and VGG16
weights from their original public sources, checks hashes and assembles Uni3R's
split checkpoint. It does not copy checkpoints from the original development
server. The setup script links `checkpoints`, `data`, `.cache`, `outputs`, `logs`
and `test_results` to EFS storage and refuses to replace existing nonempty folders.

The LSeg submodule needs `patches/lang-seg-reproduction.patch` for the existing
reproduction fixes. The setup script applies it idempotently. Other submodules
use the pinned upstream commits. The old workstation's empty rasterizer worktree
is not a source change and is not propagated as deleted source files.

Generated instance fixtures and diagnostic/student checkpoints are excluded from
Git. They are local experiment products, not publicly downloadable official
weights. Recreate them with their documented procedures when needed; cloning does
not recreate those results. See `instance_A/README.md` and `instance_B/README.md`.

This deployment does not claim a working CUDA training environment. Dependencies
still need to be installed for the target GPU/CUDA configuration before running
training or evaluation. Source `scripts/env_daxiang.sh` first so package caches
and environments stay on EFS. Full author training data has separate access and
processing requirements documented in `AUTHOR_DATASETS.md`.

To also download the public ScanNet evaluation archive, omit `--weights-only`.
Use `--views 8 16` only when those additional official checkpoints are needed.
