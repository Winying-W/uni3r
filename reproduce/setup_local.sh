#!/usr/bin/env bash
set -euo pipefail
umask 002
REPO=$(cd "$(dirname "$0")/.." && pwd)
export UNI3R_STORAGE=/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export UNI3R_PROXY=${UNI3R_PROXY:-http://127.0.0.1:37890}
export HTTPS_PROXY="$UNI3R_PROXY"
export HTTP_PROXY="$UNI3R_PROXY"
export HF_HOME="$UNI3R_STORAGE/cache/huggingface"
export TORCH_HOME="$UNI3R_STORAGE/cache/torch"
export XDG_CACHE_HOME="$UNI3R_STORAGE/cache"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
BASE=/file_system/vepfs/intern/haozhe.jia/miniforge3/envs/fastw/bin
ENV="$UNI3R_STORAGE/venv"
cd "$REPO"
exec 9>"$REPO/reproduce/.setup.lock"
flock -w 3600 9
if [ ! -x "$ENV/bin/python" ]; then "$BASE/python" -m venv --system-site-packages "$ENV"; fi
export PATH="$ENV/bin:$CUDA_HOME/bin:$BASE:$PATH"
python -m pip install -r reproduce/requirements.txt
python reproduce/prepare_sources.py
python reproduce/apply_vendor_patches.py
python -m pip install --no-build-isolation "$UNI3R_STORAGE/sources/simple-knn" "$UNI3R_STORAGE/sources/CLIP"
python test.py --help
python reproduce/prepare_assets.py
