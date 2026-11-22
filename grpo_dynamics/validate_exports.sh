#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
source "${GRPO_MACHINE_CONFIG:-$root/machine/local.env}"
run="$root/runs/${1:?provide run ID}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root:$root/verl" CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1
IFS=, read -ra cards <<< "$GRPO_VISIBLE_GPUS"
for gpu in "${cards[@]}"; do
    [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]] ||
        { echo "GPU $gpu is occupied" >&2; exit 2; }
done
baseline=()
for step in $(seq 0 10 100); do
    printf -v tag 'step_%03d' "$step"
    source_model="$run/exports/$tag"
    [[ -s "$source_model/checkpoint_manifest.json" ]] || exit 2
    dest="$run/validation/$tag"
    mkdir -p "$dest"
    pids=()
    files=()
    for rank in "${!cards[@]}"; do
        part="$dest/rank_$rank.parquet"
        files+=("$part")
        if [[ ! -s "$part" ]]; then
            CUDA_VISIBLE_DEVICES="${cards[$rank]}" "$GRPO_PYTHON" -m grpo_dynamics.evaluate generate \
                --mode validate --rank "$rank" --world-size "${#cards[@]}" \
                --model "$source_model" --data "$GRPO_DATA/validation.parquet" \
                --output "$part" > "$dest/rank_$rank.log" 2>&1 &
            pids+=("$!")
        fi
    done
    rc=0
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    (( rc == 0 )) || exit "$rc"
    "$GRPO_PYTHON" -m grpo_dynamics.evaluate summarize --mode validate --inputs "${files[@]}" \
        --output "$dest/summary.json" "${baseline[@]}"
    if [[ "$step" == 0 ]]; then
        baseline=(--baseline "${files[@]}")
    fi
done
