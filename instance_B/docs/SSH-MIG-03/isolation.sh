# Source this B-only environment isolation file before installation or tests.
export B_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$B_ROOT/.cache/pycache"
export TMPDIR="$B_ROOT/.tmp" TMP="$B_ROOT/.tmp" TEMP="$B_ROOT/.tmp"
export XDG_CACHE_HOME="$B_ROOT/.cache" XDG_DATA_HOME="$B_ROOT/.runtime/xdg-data"
export XDG_CONFIG_HOME="$B_ROOT/.runtime/xdg-config" XDG_STATE_HOME="$B_ROOT/.runtime/xdg-state"
export UV_PYTHON_INSTALL_DIR="$B_ROOT/.runtime/python" UV_PYTHON_BIN_DIR="$B_ROOT/.runtime/bin"
export UV_CACHE_DIR="$B_ROOT/.cache/uv" UV_TOOL_DIR="$B_ROOT/.tools/uv-tools" UV_TOOL_BIN_DIR="$B_ROOT/.tools/bin"
export UV_CREDENTIALS_DIR="$B_ROOT/.runtime/uv-credentials" UV_NO_CONFIG=1 UV_NO_PROGRESS=1
export UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=4
export PIP_CACHE_DIR="$B_ROOT/.cache/pip" PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
export TORCH_HOME="$B_ROOT/.cache/torch" TORCH_EXTENSIONS_DIR="$B_ROOT/.cache/torch_extensions"
export HF_HOME="$B_ROOT/.cache/huggingface" MPLCONFIGDIR="$B_ROOT/.cache/matplotlib"
export TRITON_CACHE_DIR="$B_ROOT/.cache/triton" CUDA_CACHE_PATH="$B_ROOT/.cache/cuda"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTHONHOME PYTHONPATH
