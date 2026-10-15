#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$UNI3R_ROOT"
# Only publish datasets here after checking the author's release, scene list,
# and preprocessing. The earlier public mirrors remain useful for smoke tests.
SCANNET_ROOT="${SCANNET_ROOT:-$DATA_ROOT/author_processed/scannet}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-$DATA_ROOT/author_processed/scannetpp}"
for dataset_root in "$SCANNET_ROOT" "$SCANNETPP_ROOT"; do
    if [[ ! -d "$dataset_root" ]]; then
        echo "Author-matched training data is not ready: $dataset_root" >&2
        echo "See $UNI3R_ROOT/AUTHOR_DATASETS.md for source checks and remaining requirements." >&2
        exit 2
    fi
done
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" train.py \
    --train_dataset "5_000 @ Scannet(split='train', ROOT='$SCANNET_ROOT', resolution=(256, 256), num_views=2) + 5_000 @ Scannetpp(split='train', ROOT='$SCANNETPP_ROOT', resolution=(256, 256), num_views=2)" \
    --test_dataset "100 @ Scannet(split='val', ROOT='$SCANNET_ROOT', resolution=(256, 256), seed=777, num_views=2)" \
    --train_criterion 'GaussianLoss()' --test_criterion 'GaussianLoss()' \
    --lr 0.0001 --min_lr 1e-6 --warmup_epochs 10 --epochs 100 \
    --batch_size "${BATCH_SIZE:-1}" --accum_iter "${ACCUM_ITER:-1}" \
    --num_workers "${NUM_WORKERS:-4}" --save_freq 1 --keep_freq 50 --eval_freq 1 \
    --output_dir "${SAVE_DIR:-checkpoints/train_2views}" "$@"
