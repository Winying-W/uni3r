#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$UNI3R_ROOT"
views="${1:-2}"
IFS=',' read -ra gpu_ids <<< "${GPU_IDS:-0,1,4,5}"
count="${#gpu_ids[@]}"
result_root="${RESULTS_DIR:-test_results/${views}views_full}"
mkdir -p "$result_root" logs
expected_samples="${EXPECTED_SAMPLES:-$(python - "$views" "${SCANNET_TEST_ROOT:-$DATA_ROOT/scannet_test}" <<'PY'
import sys
from uni3r.utils.path_manager import init_all_submodules
init_all_submodules()
from uni3r.datasets.testdata import TestDataset
print(len(TestDataset(ROOT=sys.argv[2], split='test', resolution=(256, 256), seed=777, num_views=int(sys.argv[1]))))
PY
)}"
pids=()
shards=()
for ((i=0; i<count; i++)); do
    shard="$result_root/shard_$i"
    shards+=("$shard")
    CUDA_VISIBLE_DEVICES="${gpu_ids[$i]}" RESULTS_DIR="$shard" \
        bash scripts/evaluate_local.sh "$views" --shard_count "$count" --shard_index "$i" \
        > "logs/eval_${views}views_shard_$i.log" 2>&1 &
    pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$result_root/launcher_pids.txt"
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if ((failed)); then echo 'An evaluation shard failed; inspect logs.' >&2; exit 1; fi
python scripts/merge_evaluation.py "${shards[@]}" \
    --expected-samples "$expected_samples" --output "$result_root/metrics.json"
