# Daxiang deployment

The code checkout belongs at `/file_system/vepfs/intern/ying.wu/project/Uni3R`.
Weights, datasets, caches, temporary files and environments belong on
`/file_system/efs/checkpoint/intern/ying.wu`; project links point to that storage.

```bash
mkdir -p /file_system/vepfs/intern/ying.wu/project
cd /file_system/vepfs/intern/ying.wu/project
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

Use `source scripts/env_daxiang.sh` for the Uni3R environment and
`source scripts/env_teacher_daxiang.sh` for the separate SAM2/GroundingDINO
environment. Both environments and their package caches live on EFS. The target
L20 GPUs use CUDA architecture 8.9. Installation and validation logs are in the
EFS `logs/` directory; a downloaded model alone is not proof of a working runtime.

The current project direction is `PROJECT_PLAN.md`. The older A/B implementation
and design records remain available as a utility library, not a replacement for
that current plan. External sources are pinned to the versions used previously:

| Source | Revision |
| --- | --- |
| facebookresearch/sam2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| WU-CVGL/SIU3R | `8ea80166be76854f938e90521f1a5b688b755c87` |
| PKU-EPIC/MaskClustering | `eb2d41c2267cde21966f4340c37b8cf94b05c23c` |

`reproduce/migration_records/` preserves the split lists, teacher comparison,
baseline scores and exact processed-scene inventories. Runtime copies are under
`data/splits/` and `test_results/`. Large prediction tensors and images are not
part of these small score records.

The requested data consists of the existing `scannet_processed` scene inventory,
219 ScanNet++ DSLR scenes, 312 SIU3R test scans (extracted data only), and the 41
SIU3R training archives selected for validation. `scannet_processed_heldout`,
ScanNet++ iPhone data and 8/16-view checkpoints are excluded. These existing
processed training mirrors are distinct from full author training data.

```bash
python3 scripts/download_teacher_weights.py
python3 scripts/download_hf_training_data.py
python3 scripts/download_siu3r_data.py --kind test
python3 scripts/download_siu3r_data.py --kind validation
python scripts/prepare_scannet_hf.py --workers 4 \
  --scene-list reproduce/migration_records/scannet_processed_scenes.txt
python scripts/prepare_scannetpp_hf.py --workers 4
```

The download commands accept `HF_ENDPOINT` and standard curl proxy environment
variables when the development host cannot reach Hugging Face directly. An SSH
download tunnel is session-specific; it is not a permanent runtime dependency.

To also download the public ScanNet evaluation archive, omit `--weights-only`.
Use `--views 8 16` only when those additional official checkpoints are needed.
