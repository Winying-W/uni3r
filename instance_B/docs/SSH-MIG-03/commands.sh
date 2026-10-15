# Executed workflow, from project root; all writes confined to instance_B.
# Tool bootstrap: official uv 0.12.7 wheel; see tool_source.json and download_official.py.
source instance_B/docs/SSH-MIG-03/isolation.sh
instance_B/.tools/uv python list --only-downloads --all-versions --output-format json 3.10
instance_B/.tools/uv python install 3.10.22 --no-bin # failed: no distribution in pinned uv catalogue
instance_B/.tools/uv python install 3.10.21 --no-bin
instance_B/.tools/uv venv --python "$B_ROOT/.runtime/python/cpython-3.10.21-linux-x86_64-gnu/bin/python3.10" "$B_ROOT/.venv"
# Initial direct-URL compile was stopped; retained in resolve.log and requirements_initial.in.
# Official torch URL/hash/size are in torch_official_source.json.
python3 -I -B -S instance_B/docs/SSH-MIG-03/download_official.py 'https://download-r2.pytorch.org/whl/cu121/torch-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl' 'instance_B/.cache/wheelhouse/torch-2.1.0+cu121-cp310-cp310-linux_x86_64.whl' 2200630334 0d4e8c52a1fcf5ed6cfc256d9a370fcf4360958fc79d0b08a51d55e70914df46
instance_B/.tools/uv pip install --python "$B_ROOT/.venv/bin/python" --only-binary :all: --index-url https://pypi.org/simple -r instance_B/docs/SSH-MIG-03/requirements-base.in -c instance_B/docs/SSH-MIG-03/constraints.txt


# Exact successful local-wheel installation and lock-generation commands:
"$B_ROOT/.tools/uv" pip install --python "$B_ROOT/.venv/bin/python" --only-binary :all: --index-url https://pypi.org/simple "$B_ROOT"/.cache/wheelhouse/numpy-1.26.4-*.whl "$B_ROOT"/.cache/wheelhouse/scipy-1.15.3-*.whl pytest==8.3.5 -c "$B_ROOT/docs/SSH-MIG-03/constraints.txt"
"$B_ROOT/.tools/uv" pip compile "$B_ROOT/docs/SSH-MIG-03/requirements.in" -c "$B_ROOT/docs/SSH-MIG-03/constraints.txt" --python "$B_ROOT/.venv/bin/python" --no-build --index-url https://pypi.org/simple --find-links "$B_ROOT/.cache/wheelhouse" --generate-hashes --no-header -o "$B_ROOT/docs/SSH-MIG-03/requirements.lock"
"$B_ROOT/.tools/uv" pip install --python "$B_ROOT/.venv/bin/python" --offline --no-deps --no-build "$B_ROOT/.cache/wheelhouse/torch-2.1.0+cu121-cp310-cp310-linux_x86_64.whl"

"$B_ROOT/.tools/uv" pip sync --python "$B_ROOT/.venv/bin/python" --offline --no-index --find-links "$B_ROOT/.cache/wheelhouse" --require-hashes --only-binary :all: "$B_ROOT/docs/SSH-MIG-03/requirements.lock"
"$B_ROOT/.venv/bin/python" -B "$B_ROOT/docs/SSH-MIG-03/run_cpu_regression.py"
