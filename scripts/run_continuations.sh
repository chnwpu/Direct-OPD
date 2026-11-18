#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 T0|T1|S0 /absolute/path/to/prefixes.jsonl" >&2
  exit 2
fi
model_name=$1
prefixes=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "${model_name}" in
  T0) model_path="${repo_root}/models/DeepSeek-R1-Distill-Qwen-1.5B"; weight=models/DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors ;;
  T1) model_path="${repo_root}/models/JustRL-DeepSeek-1.5B"; weight=models/JustRL-DeepSeek-1.5B/model.safetensors ;;
  S0) model_path="${repo_root}/models/Qwen3-1.7B"; weight=models/Qwen3-1.7B/model-00001-of-00002.safetensors ;;
  *) echo "unknown model: ${model_name}" >&2; exit 2 ;;
esac
if [ ! -f "${prefixes}" ]; then
  echo "missing frozen prefixes: ${prefixes}" >&2
  exit 2
fi
prefixes=$(realpath "${prefixes}")
run_id=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)"}
run_dir=${RUN_ROOT:-"${repo_root}/runs/horizon-fidelity/continuations"}/$(echo "${model_name}" | tr '[:upper:]' '[:lower:]')/${run_id}
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
sha256sum "${prefixes}" > "${run_dir}/prefixes.sha256"
python3 "${repo_root}/scripts/verify_horizon_inputs.py" \
  --report "${run_dir}/input_verification.json" \
  --only "${weight}" \
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
  echo "model_name=${model_name}"
  echo "model_path=${model_path}"
  echo "prefixes=${prefixes}"
  echo "seed=42"
  echo "continuations_per_prefix=8"
  echo "temperature=0.7"
  echo "top_p=0.95"
  echo "maximum_total_response_tokens=31744"
  echo "gpu_index=${gpu_index}"
} > "${run_dir}/resolved_parameters.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/started_utc.txt"
cd "${repo_root}"
"${python_bin}" scripts/run_prefix_continuations.py "${prefixes}" \
  --model "${model_path}" --model-name "${model_name}" \
  --output "${run_dir}/continuations.jsonl" \
  2>&1 | tee "${run_dir}/generation.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/finished_utc.txt"
