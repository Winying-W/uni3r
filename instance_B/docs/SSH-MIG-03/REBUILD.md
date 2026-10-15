# B environment rebuild and activation

Future rebuild requires its own authorization. Use a fresh B environment; never overwrite old `.venv`, runtime, reports or tests. Restore the approved project checkout and existing B DPT initialization independently. No weights are included in these documents.

1. Copy this lightweight docs directory and the B-local `.gitignore`. Run Bash from the new Uni3R root. `source instance_B/docs/SSH-MIG-03/isolation.sh` derives B_ROOT from this file location and confines caches/temp to B. Create B's `.tools`, `.runtime`, `.cache`, `.tmp` directories if absent.
2. Bootstrap uv 0.12.7 from the official PyPI wheel URL/size/SHA256 in `tool_source.json`. Use `download_official.py URL B_CACHE_PATH SIZE SHA256 524288` with system Python 3.11+ (stdlib only). Validate the SHA256 before using zipfile to extract only the wheel member ending `/scripts/uv` to the new `$B_ROOT/.tools/uv`, with exclusive creation and executable permission. Do not run a downloaded installer or modify shell startup files.
3. Install the exact runtime and create the isolated environment:

```bash
source instance_B/docs/SSH-MIG-03/isolation.sh
"$B_ROOT/.tools/uv" python install 3.10.21 --no-bin
test ! -e "$B_ROOT/.venv" || exit 1
"$B_ROOT/.tools/uv" venv --python "$B_ROOT/.runtime/python/cpython-3.10.21-linux-x86_64-gnu/bin/python3.10" "$B_ROOT/.venv"
```

The original Python 3.10.22 was not available in the pinned uv catalogue; the actual failed attempt is retained. `python_candidates.json` records the selected Astral python-build-standalone official URL. This is a recorded patch/build difference, not an exact binary copy of the old environment.

4. Install the complete pinned dependency set, verifying hashes:

```bash
"$B_ROOT/.tools/uv" pip sync --python "$B_ROOT/.venv/bin/python" --require-hashes --only-binary :all: --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu121/torch/ instance_B/docs/SSH-MIG-03/requirements.lock
"$B_ROOT/.tools/uv" pip check --python "$B_ROOT/.venv/bin/python"
```

For slow transfers, use the exact official URL, size and SHA256 in the source JSON files with `download_official.py` to populate `$B_ROOT/.cache/wheelhouse`, then replace the network find-links argument with that local wheelhouse. Torch's size is 2200630334 bytes and its official SHA256 is in `torch_official_source.json`. Do not execute partial files. Download helper validates range lengths and assembled SHA256. Sources: official PyPI/files.pythonhosted.org, PyTorch cu121 index/download-r2.pytorch.org, and Astral Python distributions. See https://docs.astral.sh/uv/reference/storage/ and https://docs.pytorch.org/get-started/previous-versions/ .

5. Activate for CPU-only work:

```bash
source instance_B/docs/SSH-MIG-03/isolation.sh
source "$B_ROOT/.venv/bin/activate"
# Use the absolute interpreter in all test commands:
"$B_ROOT/.venv/bin/python" -B "$B_ROOT/docs/SSH-MIG-03/run_cpu_regression.py"
```

The driver first verifies versions, CPU tensor math, NumPy conversions and core imports, and checks dependency consistency. Pytest collects the five original modules. The unchanged `instance_B.run_b02_checks` unit/integration modes execute the original 76 tests. Future runs must use newly authorized output names in this B-owned wrapper: existing outputs deliberately stop execution instead of overwriting evidence.

Only existing B DPT initialization may be loaded for the approved synthetic CPU tests. Roundtrip state serialization stays in memory. No optimizer step, training checkpoint, full Uni3R, Gaussian rendering, A/C integration or GPU computation is included. Keep CUDA_VISIBLE_DEVICES empty, bytecode disabled, and two CPU math threads. Activation alone is insufficient; source the isolation file first.

`requirements.in` gives four direct dependencies, `constraints.txt` preserves relevant old runtime pins plus explicitly selected pytest helpers, and `requirements.lock` is the full hashed dependency lock. Unused constraints do not cause packages to be installed. Torchvision/Torchaudio and the full Uni3R stack are intentionally absent because original B imports do not need them.
