#!/usr/bin/env bash
set -euo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
export UNI3R_STORAGE=/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R
REPO=/file_system/vepfs/intern/haozhe.jia/projects/FastWAM/3DGS/Uni3R
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$UNI3R_STORAGE/cache/huggingface"
export TORCH_HOME="$UNI3R_STORAGE/cache/torch"
export XDG_CACHE_HOME="$UNI3R_STORAGE/cache"
export UNI3R_LSEG_CHECKPOINT="$UNI3R_STORAGE/checkpoints/demo_e200.ckpt"
export CUDA_HOME="$UNI3R_STORAGE/cuda-12.9"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=8
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN="$UNI3R_STORAGE/runs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
chmod 777 "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
trap 'rc=$?; echo "$rc" > "$RUN/exit_code"' EXIT
cd "$REPO"
nvidia-smi
BASE=/file_system/vepfs/intern/haozhe.jia/miniforge3/envs/fastw/bin/python
ENV="$UNI3R_STORAGE/venv"
# Serialize preparation against a local preparation process.
exec 9>"$REPO/reproduce/.setup.lock"
flock -w 3600 9
if [ ! -x "$ENV/bin/python" ]; then "$BASE" -m venv --system-site-packages "$ENV"; fi
export PATH="$ENV/bin:$CUDA_HOME/bin:/file_system/vepfs/intern/haozhe.jia/miniforge3/envs/fastw/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
python -m pip install -r reproduce/requirements.txt
python reproduce/prepare_sources.py
python -m pip install --no-build-isolation "$UNI3R_STORAGE/sources/simple-knn"
python -m pip install --no-build-isolation "$UNI3R_STORAGE/sources/CLIP"
(cd submodules/dust3r/croco/models/curope && python setup.py build_ext --inplace)
python reproduce/prepare_assets.py
python -m pip freeze > "$RUN/packages.txt"
git diff > "$RUN/code.patch"
git rev-parse HEAD > "$RUN/commit.txt"
python test.py --help
flock -u 9
if [ "${UNI3R_PREPARE_ONLY:-0}" = 1 ]; then exit 0; fi
DATA=$(cat "$UNI3R_STORAGE/data_root.txt")
pids=()
for entry in 0:2 1:8 2:16; do
    gpu=${entry%:*}; views=${entry#*:}
    mkdir -p "$RUN/${views}views"
    (
        # These are three independent full evaluations, not distributed shards.
        unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
        export CUDA_VISIBLE_DEVICES="$gpu"
        python test.py \
          --test_dataset "TestDataset(split='test', ROOT='$DATA', resolution=(256, 256), seed=777, num_views=$views)" \
          --test_criterion "TestLoss(pose_align_steps=100, num_views=$views)" \
          --pretrained "$UNI3R_STORAGE/checkpoints/${views}views.pth" \
          --lseg_pretrained "$UNI3R_LSEG_CHECKPOINT" \
          --num_workers 4 --test_results_dir "$RUN/${views}views"
    ) > "$RUN/${views}views/eval.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
