#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 FIXED_TRAJECTORY_DIR H2_TRAIN_DIR H6_TRAIN_DIR H4_TRAIN_DIR" >&2
  exit 2
fi

frozen=$(realpath "$1")
h2=$(realpath "$2")
h6=$(realpath "$3")
h4=$(realpath "$4")
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo}"
run_id=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)"}
run_dir=${RUN_ROOT:-"${repo}/runs/horizon-fidelity/analysis"}/${run_id}
mkdir -p "$(dirname "${run_dir}")"
mkdir "${run_dir}"
mkdir "${run_dir}/logs"
python_bin=${PYTHON_BIN:-"${repo}/.envs/direct-opd/bin/python"}
export PYTHONNOUSERSITE=1 PYTHONPATH="${repo}/verl"

git rev-parse HEAD > "${run_dir}/code_commit.txt"
git diff > "${run_dir}/uncommitted.patch"
"${python_bin}" -m pip freeze > "${run_dir}/pip_freeze.txt"
sha256sum "${frozen}/trajectories.jsonl" "${frozen}/prefixes.jsonl" > "${run_dir}/frozen_inputs.sha256"
python3 scripts/verify_horizon_inputs.py --report "${run_dir}/input_verification.json"
python3 scripts/select_idle_gpus.py --snapshot "${run_dir}/gpu_selection.json" --batch-size 256 > /dev/null
mapfile -t gpus < <(python3 -c 'import json,sys; print(*json.load(open(sys.argv[1]))["idle_indices"], sep="\n")' "${run_dir}/gpu_selection.json")

checkpoint_at() {
  local run=$1 step=$2
  local matches=("${run}"/checkpoints/*/global_step_"${step}"/actor)
  if [ "${#matches[@]}" -ne 1 ] || [ ! -d "${matches[0]}" ]; then
    echo "missing or ambiguous step ${step} actor checkpoint under ${run}" >&2
    return 1
  fi
  realpath "${matches[0]}"
}

merge_actor() {
  local checkpoint=$1 model=$2
  CUDA_VISIBLE_DEVICES= "${python_bin}" -m verl.model_merger merge --backend fsdp \
    --use_cpu_initialization --local_dir "${checkpoint}" --target_dir "${model}"
}

mkdir "${run_dir}/exported"
for horizon in h2 h6 h4; do
  train_dir=${!horizon}
  checkpoint=$(checkpoint_at "${train_dir}" 40)
  printf '%s=%s\n' "${horizon}" "${checkpoint}" >> "${run_dir}/checkpoints.txt"
  merge_actor "${checkpoint}" "${run_dir}/exported/${horizon}" > "${run_dir}/logs/merge_${horizon}.log" 2>&1
  "${python_bin}" scripts/summarize_training_fidelity.py "${train_dir}/rollouts" \
    --output "${run_dir}/${horizon}_training_fidelity.json"
done

pids=()
wait_batch() {
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  pids=()
  return "${failed}"
}
submit() {
  local name=$1
  shift
  if [ "${#pids[@]}" -eq "${#gpus[@]}" ]; then
    wait_batch
  fi
  local gpu=${gpus[${#pids[@]}]}
  CUDA_VISIBLE_DEVICES="${gpu}" "$@" > "${run_dir}/logs/${name}.log" 2>&1 &
  pids+=("$!")
}

for name in S0 H2 H4 H6; do
  model="${run_dir}/exported/${name,,}"
  if [ "${name}" = S0 ]; then
    model="${repo}/models/Qwen3-1.7B"
  fi
  submit "forward_${name}" "${python_bin}" scripts/teacher_forced_stats.py "${frozen}/trajectories.jsonl" \
    --model "${model}" --model-name "${name}" --output-dir "${run_dir}/forwards/${name}"
done
wait_batch

submit forward_T0 "${python_bin}" scripts/teacher_forced_stats.py "${frozen}/trajectories.jsonl" \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B --model-name T0 --output-dir "${run_dir}/forwards/T0" \
  --student-stats-dir "${run_dir}/forwards"
submit forward_T1 "${python_bin}" scripts/teacher_forced_stats.py "${frozen}/trajectories.jsonl" \
  --model models/JustRL-DeepSeek-1.5B --model-name T1 --output-dir "${run_dir}/forwards/T1" \
  --student-stats-dir "${run_dir}/forwards"
for name in T0 T1 S0; do
  case "${name}" in
    T0) model=models/DeepSeek-R1-Distill-Qwen-1.5B ;;
    T1) model=models/JustRL-DeepSeek-1.5B ;;
    S0) model=models/Qwen3-1.7B ;;
  esac
  submit "continuation_${name}" "${python_bin}" scripts/run_prefix_continuations.py "${frozen}/prefixes.jsonl" \
    --model "${model}" --model-name "${name}" --output "${run_dir}/continuations/${name}.jsonl"
done
wait_batch

"${python_bin}" scripts/aggregate_teacher_shift.py "${frozen}/trajectories.jsonl" \
  --stats-dir "${run_dir}/forwards" --output "${run_dir}/teacher_shift.json"
MPLCONFIGDIR="${run_dir}/matplotlib" python3 scripts/plot_teacher_shift.py \
  "${run_dir}/teacher_shift.json" --output "${run_dir}/figure8.pdf"
"${python_bin}" scripts/summarize_prefix_continuations.py \
  --t0 "${run_dir}/continuations/T0.jsonl" --t1 "${run_dir}/continuations/T1.jsonl" \
  --s0 "${run_dir}/continuations/S0.jsonl" --shift-stats-dir "${run_dir}/forwards" \
  --output "${run_dir}/prefix_fidelity.json"

"${python_bin}" scripts/select_token_matched.py "${h2}/run_summary.json" "${h6}/run_summary.json" \
  --metric generated --output "${run_dir}/token_matched_h2_h6.json"
matched_step=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["comparison"]["step"])' "${run_dir}/token_matched_h2_h6.json")
h6_model="${run_dir}/exported/h6"
if [ "${matched_step}" -ne 40 ]; then
  h6_model="${run_dir}/exported/h6_step_${matched_step}"
  checkpoint=$(checkpoint_at "${h6}" "${matched_step}")
  merge_actor "${checkpoint}" "${h6_model}" > "${run_dir}/logs/merge_h6_matched.log" 2>&1
fi
submit matched_h2 "${python_bin}" scripts/evaluate_horizon_checkpoint.py \
  --model "${run_dir}/exported/h2" --output-dir "${run_dir}/matched_h2_step_40"
submit matched_h6 "${python_bin}" scripts/evaluate_horizon_checkpoint.py \
  --model "${h6_model}" --output-dir "${run_dir}/matched_h6_step_${matched_step}"
wait_batch
"${python_bin}" scripts/summarize_horizon_accuracy.py \
  --h2 "${h2}" --h4 "${h4}" --h6 "${h6}" --analysis "${run_dir}"

python3 - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
fidelity = json.loads((root / "prefix_fidelity.json").read_text())
change = fidelity["paired_change_from_512_to_4096"]
trigger = bool(
    change and change["mean_absolute_shift_change_4096_minus_512"] > 0
    and min(change["change_95_percent_interval"][1], change["t1_change_95_percent_interval"][1]) < 0
)
(root / "candidate_probe_gate.json").write_text(json.dumps({
    "triggered": trigger,
    "rule": (
        "Among matched 512/4096 prefixes, mean absolute Eq. 20 shift increases and "
        "the 95% conditional interval for change in F or T1 success lies below zero"
    ),
    "paired_change": change,
}, indent=2) + "\n")
PY
if python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["triggered"])' \
  "${run_dir}/candidate_probe_gate.json"; then
  submit select_actions "${python_bin}" scripts/select_candidate_probes.py \
    "${frozen}/trajectories.jsonl" "${run_dir}/forwards" --output "${run_dir}/action_probes.jsonl"
  wait_batch
  submit probe_actions "${python_bin}" scripts/run_candidate_probes.py \
    "${run_dir}/action_probes.jsonl" --output-dir "${run_dir}/candidate_values"
  wait_batch
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/finished_utc.txt"
