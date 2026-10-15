#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$UNI3R_ROOT"
if [[ ! -x "$UNI3R_ENV/bin/python" ]]; then
    "${CONDA_EXE:-/root/miniconda3/bin/conda}" create -y -p "$UNI3R_ENV" python=3.10
fi
export UV_CACHE_DIR="$UNI3R_ROOT/.cache/uv"
export UV_LINK_MODE=copy
index="${UNI3R_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
uv pip install --python "$UNI3R_ENV/bin/python" --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.1.0' 'torchvision==0.16.0'
uv pip install --python "$UNI3R_ENV/bin/python" --index-url "$index" \
    -r requirements-reproduction.txt 'torch==2.1.0' 'torchvision==0.16.0'

clone_source() {
    local url="$1" directory="$2" revision="$3"
    if [[ ! -d "$directory" ]]; then
        git clone "$url" "$directory"
        git -C "$directory" checkout "$revision"
    fi
}
clone_source https://github.com/openai/CLIP.git submodules/CLIP d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
clone_source https://github.com/camenduru/simple-knn.git submodules/simple-knn 60f461f4a56b7967e5d8045bf92f8c33f36976d0
clone_source https://github.com/facebookresearch/pytorch3d.git submodules/pytorch3d 89653419d0973396f3eff1a381ba09a07fffc2ed
uv pip install --python "$UNI3R_ENV/bin/python" --no-build-isolation --no-deps \
    submodules/CLIP submodules/simple-knn submodules/pytorch3d
(cd submodules/dust3r/croco/models/curope && python setup.py build_ext --inplace)
python -m pip check
