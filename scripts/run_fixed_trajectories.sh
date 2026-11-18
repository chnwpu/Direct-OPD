#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)"}
run_dir=${RUN_ROOT:-"${repo_root}/runs/horizon-fidelity/fixed-trajectories"}/${run_id}
if [ -e "${run_dir}" ]; then
  echo "run directory already exists: ${run_dir}" >&2
  exit 1
fi
mkdir -p "${run_dir}"
python_bin=${PYTHON_BIN:-"${repo_root}/.envs/direct-opd/bin/python"}
export PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}/verl"
git -C "${repo_root}" rev-parse HEAD > "${run_dir}/code_commit.txt"
git -C "${repo_root}" diff > "${run_dir}/uncommitted.patch"
"${python_bin}" -m pip freeze > "${run_dir}/pip_freeze.txt" 2>&1
uname -a > "${run_dir}/system.txt"
python3 "${repo_root}/scripts/verify_horizon_inputs.py" \
  --report "${run_dir}/input_verification.json" \
  --only \
  models/Qwen3-1.7B/model-00001-of-00002.safetensors \
  models/Qwen3-1.7B/model-00002-of-00002.safetensors \
  datasets/eval/aime24.parquet datasets/eval/aime25.parquet

python3 "${repo_root}/scripts/select_idle_gpus.py" \
  --snapshot "${run_dir}/gpu_selection.json" --batch-size 256 > /dev/null
gpu_index=$(python3 - "${run_dir}/gpu_selection.json" "${ANALYSIS_GPU:-}" <<'PY'
import json
import sys
snapshot = json.load(open(sys.argv[1]))
idle = snapshot["idle_indices"]
requested = sys.argv[2]
if requested:
    index = int(requested)
    if index not in idle:
        raise SystemExit(f"GPU {index} is not idle")
else:
    extras = snapshot["idle_but_unselected_indices"]
    index = extras[0] if extras else idle[-1]
print(index)
PY
)
export CUDA_VISIBLE_DEVICES="${gpu_index}"
{
  echo "model=Qwen/Qwen3-1.7B"
  echo "seed=42"
  echo "trajectory_count=64"
  echo "maximum_response_length=16384"
  echo "temperature=1.0"
  echo "top_p=1.0"
  echo "gpu_index=${gpu_index}"
} > "${run_dir}/resolved_parameters.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/started_utc.txt"
cd "${repo_root}"
"${python_bin}" scripts/generate_fixed_trajectories.py \
  --output "${run_dir}/trajectories.jsonl" \
  2>&1 | tee "${run_dir}/generation.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/finished_utc.txt"
