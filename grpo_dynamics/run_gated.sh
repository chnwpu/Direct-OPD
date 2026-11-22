#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
source "${GRPO_MACHINE_CONFIG:-$root/machine/local.env}"
suffix="${1:?provide run suffix}"
[[ "$suffix" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
export PYTHONNOUSERSITE=1 PYTHONPATH="$root:$root/verl"
if [[ ! -s "$GRPO_DATA/preflight_summary.json" ]]; then
    echo "Awaiting current DeepMath preflight."
    deadline=$((SECONDS + 7200))
    while [[ ! -s "$GRPO_DATA/preflight_summary.json" && $SECONDS -lt $deadline ]]; do
        pgrep -f '^bash grpo_dynamics/preflight.sh ' >/dev/null || { echo "Preflight exited without a summary." >&2; exit 2; }
        sleep 15
    done
fi
"$GRPO_PYTHON" - "$GRPO_DATA/preflight_summary.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print("Preflight group ratios:", {k:v for k,v in p.items() if k.endswith("group_ratio")}, flush=True)
if not p["gate_passed"]:
    raise SystemExit("Difficulty gate failed. Formal training has not started.")
PY

export GRPO_SMOKE_RUN="$root/runs/smoke-$suffix"
export GRPO_PROFILE_RUN="$root/runs/profile-$suffix"
formal="qwen3-1p7b-grpo-$suffix"
bash grpo_dynamics/train.sh smoke "smoke-$suffix"
CUDA_VISIBLE_DEVICES= bash grpo_dynamics/postprocess.sh "smoke-$suffix" 0 1
bash grpo_dynamics/train.sh smoke-resume "smoke-$suffix"
"$GRPO_PYTHON" - "$GRPO_SMOKE_RUN" <<'PY'
from pathlib import Path
import sys,json
from grpo_dynamics.record import atomic_json
p=Path(sys.argv[1])
log=(p/"logs/trainer-smoke-resume.log").read_text()
assert "Setting global step to 1" in log
assert (p/"checkpoints/global_step_1/driver_rng.pt").is_file()
assert (p/"checkpoints/global_step_1/data.pt").is_file()
step=json.loads((p/"steps/step_002.json").read_text())
assert step["global_step"] == 2
atomic_json(p/"resume-smoke-result.json", {"reached_step":2,"loaded_optimizer_scheduler_rng_dataloader":True})
PY
"$GRPO_PYTHON" -m grpo_dynamics.gates audit --mode smoke --data "$GRPO_DATA" --model "$GRPO_MODEL" --run "$GRPO_SMOKE_RUN"
bash grpo_dynamics/train.sh profile "profile-$suffix"
"$GRPO_PYTHON" -m grpo_dynamics.gates audit --mode profile --data "$GRPO_DATA" --model "$GRPO_MODEL" --run "$GRPO_PROFILE_RUN"
bash grpo_dynamics/train.sh train "$formal"
mapfile -t checkpoints < <(seq 0 10 100)
CUDA_VISIBLE_DEVICES= bash grpo_dynamics/postprocess.sh "$formal" "${checkpoints[@]}"
bash grpo_dynamics/validate_exports.sh "$formal"
