#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export UNI3R_STORAGE=/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R
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
export PYTHONDONTWRITEBYTECODE=1
: "${UNI3R_GPU:?Select a free GPU with nvidia-smi, then set UNI3R_GPU}"
export CUDA_VISIBLE_DEVICES="$UNI3R_GPU"
cd "$REPO"
python reproduce/oracle_instance_lifting.py "$@"
