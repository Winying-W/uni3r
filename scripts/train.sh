#!/usr/bin/env bash
set -euo pipefail
# Preserve the upstream 8-GPU/batch-4 defaults and require author-matched data.
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export SAVE_DIR="${SAVE_DIR:-checkpoints/output}"
exec bash "$(dirname "$0")/train_local.sh" "$@"
