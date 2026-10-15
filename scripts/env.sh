#!/usr/bin/env bash
# Source this file before running Uni3R from any directory.
UNI3R_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNI3R_ROOT
export UNI3R_ENV="${UNI3R_ENV:-/root/miniconda3/envs/uni3r}"
export PATH="$UNI3R_ENV/bin:$PATH"
export DATA_ROOT="${DATA_ROOT:-$(dirname "$UNI3R_ROOT")/data}"
export TORCH_HOME="$UNI3R_ROOT/checkpoints/torch"
export HF_HOME="$UNI3R_ROOT/.cache/huggingface"
export MPLCONFIGDIR="$UNI3R_ROOT/.cache/matplotlib"
export UNI3R_CLIP_CACHE="$UNI3R_ROOT/checkpoints/clip"
export TORCH_EXTENSIONS_DIR="$UNI3R_ROOT/.cache/torch_extensions"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${UNI3R_CUDA_ARCH:-8.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
