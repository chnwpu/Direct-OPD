#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
source "${GRPO_MACHINE_CONFIG:-$root/machine/local.env}"
run="$root/runs/${1:?expected run ID}"
shift
export PYTHONNOUSERSITE=1 PYTHONPATH="$root:$root/verl"
exec 8>"$run/.export.lock"
flock 8
for step in "$@"; do
    printf -v tag 'step_%03d' "$step"
    dest="$run/exports/$tag"
    if [[ -s "$dest/checkpoint_manifest.json" ]]; then continue; fi
    mkdir -p "$dest"
    flags=()
    if [[ "$step" == 0 ]]; then
        for file in "$GRPO_MODEL"/*; do ln -sfn "$file" "$dest/"; done
    else
        src="$run/checkpoints/global_step_$step/actor"
        "$GRPO_PYTHON" -m verl.model_merger merge --backend fsdp --use_cpu_initialization \
            --local_dir "$src" --target_dir "$dest"
        if [[ "$step" == 1 ]]; then flags+=(--compare-original); fi
    fi
    "$GRPO_PYTHON" -m grpo_dynamics.export --run "$run" --step "$step" --model "$dest" \
        --initial-model "$GRPO_MODEL" "${flags[@]}"
done
