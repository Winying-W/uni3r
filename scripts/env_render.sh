#!/usr/bin/env bash
# Independent Python 3.9 environment for the author's ScanNet++ preprocessor.
UNI3R_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNI3R_ROOT
export UNI3R_RENDER_ENV="$UNI3R_ROOT/.cache/renderpy-env"
export PATH="$UNI3R_RENDER_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$UNI3R_ROOT/.cache/render-native/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
