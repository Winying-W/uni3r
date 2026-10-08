#!/usr/bin/env bash
set -euo pipefail
umask 002
REPO=$(cd "$(dirname "$0")/.." && pwd)
export UNI3R_STORAGE=/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R
export HF_ENDPOINT=https://huggingface.co
export UNI3R_PROXY=${UNI3R_PROXY:-http://127.0.0.1:37890}
export HTTPS_PROXY="$UNI3R_PROXY"
export HTTP_PROXY="$UNI3R_PROXY"
export HF_HOME="$UNI3R_STORAGE/cache/huggingface"
export TORCH_HOME="$UNI3R_STORAGE/cache/torch"
export XDG_CACHE_HOME="$UNI3R_STORAGE/cache"
export UNI3R_LSEG_CHECKPOINT="$UNI3R_STORAGE/checkpoints/demo_e200.ckpt"
export CUDA_HOME="$UNI3R_STORAGE/cuda-12.9"
export PATH="$UNI3R_STORAGE/venv/bin:$CUDA_HOME/bin:/file_system/vepfs/intern/haozhe.jia/miniforge3/envs/fastw/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$REPO/reproduce/build_cache"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$REPO"
if [ "${1:-}" = --prepare ]; then
    bash reproduce/setup_local.sh
    exit
fi
# Require explicit selection after checking the shared development machine.
: "${UNI3R_GPU:?Set UNI3R_GPU to a GPU index after checking nvidia-smi}"
DATA=$(cat "$UNI3R_STORAGE/data_root.txt")
VIEWS=${UNI3R_VIEWS:-"2 8 16"}
for views in $VIEWS; do
    case "$views" in 2|8|16) ;; *) echo "Unsupported view count: $views" >&2; exit 2;; esac
    test -s "$UNI3R_STORAGE/checkpoints/${views}views.pth"
done
test -s "$UNI3R_LSEG_CHECKPOINT"
EXTRA_ARGS=()
if [ -n "${UNI3R_MAX_BATCHES:-}" ]; then EXTRA_ARGS+=(--max_batches "$UNI3R_MAX_BATCHES"); fi
RUN="$UNI3R_STORAGE/runs/local-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
python -m pip freeze > "$RUN/packages.txt"
git diff > "$RUN/code.patch"
cp reproduce/lseg_vendor.patch "$RUN/lseg_vendor.patch"
# Sequential settings avoid reserving three busy development GPUs at once.
for views in $VIEWS; do
    mkdir -p "$RUN/${views}views"
    CUDA_VISIBLE_DEVICES="$UNI3R_GPU" python test.py \
      --test_dataset "TestDataset(split='test', ROOT='$DATA', resolution=(256, 256), seed=777, num_views=$views)" \
      --test_criterion "TestLoss(pose_align_steps=100, num_views=$views, fast_pose_alignment=${UNI3R_FAST_ALIGNMENT:-False})" \
      --pretrained "$UNI3R_STORAGE/checkpoints/${views}views.pth" \
      --num_workers 4 --test_results_dir "$RUN/${views}views" "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$RUN/${views}views/eval.log"
done
