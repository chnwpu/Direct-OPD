#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
source "${GRPO_MACHINE_CONFIG:-$root/machine/local.env}"
run="$root/runs/${1:?Run ID required}"
out="$run/dynamics/teacher_pilot"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root:$root/verl" CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
mkdir -p "$out"
exec 9>"$out/.lock"
flock -n 9
[[ -f "$out/bank.json" ]] || "$GRPO_PYTHON" -m grpo_dynamics.teacher_pilot freeze --run "$run" --out "$out"
cards=()
while IFS=, read -r gpu memory; do
    if (( memory < 128 )) && [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
        cards+=("$gpu")
    fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
[[ "${#cards[@]}" -gt 0 ]] || { echo "No idle GPU"; exit 2; }
pids=()
for i in "${!cards[@]}"; do
    (
        export CUDA_VISIBLE_DEVICES="${cards[$i]}"
        for ((j=i; j<=10; j+=${#cards[@]})); do
            step=$((j*10))
            printf -v tag 'step_%03d' "$step"
            if [[ ! -s "$out/${tag}_score.json" || ! -s "$out/${tag}_logp.npy" ]]; then
                "$GRPO_PYTHON" -m grpo_dynamics.teacher_pilot score --run "$run" --out "$out" --step "$step"
            fi
        done
    ) > "$out/gpu_${cards[$i]}.log" 2>&1 &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( rc == 0 )) || exit "$rc"
"$GRPO_PYTHON" -m grpo_dynamics.teacher_pilot summarize --run "$run" --out "$out"
