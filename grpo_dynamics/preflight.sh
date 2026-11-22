#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
source "${GRPO_MACHINE_CONFIG:-$root/machine/local.env}"
run_id="${1:?provide independent preflight run ID}"
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
run="$root/runs/$run_id"
mkdir -p "$run"
exec 9>"$run/.lock"
flock -n 9
export PYTHONNOUSERSITE=1 PYTHONPATH="$root:$root/verl" CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1
if [[ ! -s "$GRPO_DATA/manifest.json" ]]; then
    "$GRPO_PYTHON" -m grpo_dynamics.prepare --raw "$GRPO_DATA/raw" --out "$GRPO_DATA" --tokenizer "$GRPO_MODEL"
fi
"$GRPO_PYTHON" -m grpo_dynamics.gates verify --mode smoke --data "$GRPO_DATA" --model "$GRPO_MODEL"
mapfile -t cards < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2 < 128 {print $1}')
[[ ${#cards[@]} -gt 0 ]] || { echo "No idle GPU for preflight"; exit 2; }
pids=()
files=()
for rank in "${!cards[@]}"; do
    gpu=${cards[$rank]}
    [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || exit 2
    part="$run/rank_$rank.parquet"
    files+=("$part")
    CUDA_VISIBLE_DEVICES="$gpu" "$GRPO_PYTHON" -m grpo_dynamics.evaluate generate --mode preflight \
        --rank "$rank" --world-size "${#cards[@]}" --model "$GRPO_MODEL" \
        --data "$GRPO_DATA/preflight.parquet" --output "$part" > "$run/rank_$rank.log" 2>&1 &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( rc == 0 )) || exit "$rc"
"$GRPO_PYTHON" -m grpo_dynamics.evaluate summarize --mode preflight --inputs "${files[@]}" \
    --manifest "$GRPO_DATA/manifest.json" --output "$run/summary.json"
cp "$run/summary.json" "$GRPO_DATA/preflight_summary.json.tmp"
mv "$GRPO_DATA/preflight_summary.json.tmp" "$GRPO_DATA/preflight_summary.json"
