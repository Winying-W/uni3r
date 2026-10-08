#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PY=/file_system/efs/checkpoint/intern/haozhe.jia/Uni3R/venv/bin/python
"$PY" reproduce/assemble_checkpoint.py --views 2 --wait_seconds 7200
# The development GPUs are shared; do not launch without enough available memory.
GPU=$(/usr/bin/nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | "$PY" -c 'import sys; rows=[tuple(map(int,s.split(","))) for s in sys.stdin]; candidates=[r for r in rows if r[1]>=26000 and r[2]<60]; print(min(candidates,key=lambda r:(r[2],-r[1]))[0] if candidates else "")')
if [ -z "$GPU" ]; then echo 'Checkpoint ready; no local GPU currently has sufficient free capacity.'; exit 3; fi
printf 'Running 2-view smoke on local GPU %s\n' "$GPU"
UNI3R_GPU="$GPU" UNI3R_VIEWS=2 UNI3R_MAX_BATCHES=1 bash reproduce/run_local.sh
