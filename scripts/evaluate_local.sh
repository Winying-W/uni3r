#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$UNI3R_ROOT"
views="${1:-2}"
if [[ $# -gt 0 ]]; then shift; fi
case "$views" in 2|8|16) ;; *) echo 'Views must be 2, 8, or 16' >&2; exit 2 ;; esac
checkpoint="${CHECKPOINT:-checkpoints/output/${views}views.pth}"
test_root="${SCANNET_TEST_ROOT:-$DATA_ROOT/scannet_test}"
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
[[ -f "$test_root/selected_seqs_test.json" ]] || { echo "Missing test dataset: $test_root" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is unavailable"'
python test.py \
  --test_dataset "TestDataset(split='test', ROOT='$test_root', resolution=(256, 256), seed=777, num_views=$views)" \
  --test_criterion "TestLoss(pose_align_steps=100, num_views=$views)" \
  --pretrained "$checkpoint" \
  --num_workers "${NUM_WORKERS:-4}" \
  --test_results_dir "${RESULTS_DIR:-test_results/${views}views}" "$@"
