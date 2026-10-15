#!/usr/bin/env bash
# usage: run_shard.sh <variant: strict|align100> <shard_count> <shard_index> <gpu>
# resumable: already-written pair dirs are skipped by export_uni3r.py
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env_daxiang.sh
export UNI3R_LSEG_CHECKPOINT=$PWD/checkpoints/pretrained_models/demo_e200.ckpt UNI3R_EVAL_SKIP_DISTILLER=1
V=$1; N=$2; I=$3; G=$4
case $V in strict) STEPS=0 ;; align100) STEPS=100 ;; *) echo bad variant; exit 2 ;; esac
CUDA_VISIBLE_DEVICES=$G python scripts/siu3r_protocol/export_uni3r.py \
  --siu3r_repo "$PWD/external/SIU3R" \
  --root "$DATA_ROOT/siu3r_protocol_root" \
  --out test_results/siu3r_protocol/uni3r_2views_$V \
  --shard_count $N --shard_index $I --pose_align_steps $STEPS
echo EXIT=$?
