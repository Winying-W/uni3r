#!/usr/bin/env bash
# Rebuild renderpy using the native dependencies prepared for this machine.
set -euo pipefail
source "$(dirname "$0")/env_render.sh"
render_native="$UNI3R_ROOT/.cache/render-native"
render_build="${RENDER_BUILD_DIR:-/tmp/uni3r-render-build}"
render_source="$UNI3R_ROOT/submodules/renderpy"
render_revision=57a9308d8d92b26b9714d8c7fa6b6103971eccd6
[[ "$(git -C "$render_source" rev-parse HEAD)" == "$render_revision" ]]
[[ -f "$render_native/lib/cmake/opencv4/OpenCVConfig.cmake" ]]
[[ -x "$UNI3R_RENDER_ENV/bin/python" ]]
if [[ ! -f "$render_build/boost/usr/include/boost/serialization/array_wrapper.hpp" ]]; then
    mkdir -p "$render_build/boost"
    dpkg-deb -x "$UNI3R_ROOT/.cache/render-build/libboost1.83-dev_1.83.0-2.1ubuntu3.2_amd64.deb" "$render_build/boost"
fi
export UV_CACHE_DIR="$UNI3R_ROOT/.cache/uv"
export CPATH="$render_native/include:$render_build/boost/usr/include"
export LIBRARY_PATH="$render_native/lib"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$render_native -DCMAKE_BUILD_RPATH=$render_native/lib -DBUILD_TESTING=OFF -DOpenGL_GL_PREFERENCE=GLVND"
"${UV_BIN:-/root/.local/bin/uv}" pip install --python "$UNI3R_RENDER_ENV/bin/python" \
    --no-build-isolation --no-deps "$render_source"
