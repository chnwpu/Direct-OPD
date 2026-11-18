#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 H2|H4|H6 [smoke|profile|train]" >&2
  exit 2
fi

case "$1" in
  H2) max_response_length=2048 ;;
  H4) max_response_length=4096 ;;
  H6) max_response_length=6144 ;;
  *) echo "unknown horizon: $1" >&2; exit 2 ;;
esac
phase=${2:-train}
case "${phase}" in
  smoke) training_steps=1; test_freq=-1; val_before_train=False ;;
  profile) training_steps=5; test_freq=-1; val_before_train=False ;;
  train) training_steps=40; test_freq=20; val_before_train=True ;;
  *) echo "unknown phase: ${phase}" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${RUN_ROOT:-"${repo_root}/runs/horizon-fidelity"}
run_id=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)"}
run_dir="${run_root}/$(echo "$1" | tr '[:upper:]' '[:lower:]')/${phase}/${run_id}"
if [ -e "${run_dir}" ]; then
  echo "run directory already exists: ${run_dir}" >&2
  exit 1
fi
mkdir -p "${run_dir}"

model_root=${MODEL_ROOT:-"${repo_root}/models"}
data_root=${DATA_ROOT:-"${repo_root}/datasets"}
python_bin=${PYTHON_BIN:-"$(command -v python3)"}
experiment_name="horizon_fidelity_$(echo "$1" | tr '[:upper:]' '[:lower:]')_${phase}_${run_id}"
git -C "${repo_root}" rev-parse HEAD > "${run_dir}/code_commit.txt"
git -C "${repo_root}" diff > "${run_dir}/uncommitted.patch"
"${python_bin}" -m pip freeze > "${run_dir}/pip_freeze.txt" 2>&1 || true
uname -a > "${run_dir}/system.txt"
{
  echo "horizon=$1"
  echo "phase=${phase}"
  echo "max_response_length=${max_response_length}"
  echo "training_steps=${training_steps}"
  echo "seed=42"
  echo "train_batch_size=64"
  echo "mini_batch_size=64"
  echo "rollout_n=4"
  echo "max_prompt_length=1024"
  echo "train_temperature=1.0"
  echo "train_top_p=1.0"
  echo "student_top_k=16"
  echo "learning_rate=1e-6"
  echo "fixed_kl_coef=1"
  echo "checkpoint_interval=5"
  echo "validation_steps=$([ "${phase}" = train ] && echo 0,20,40 || echo disabled)"
  echo "validation_samples_per_problem=32"
  echo "validation_temperature=0.7"
  echo "validation_top_p=0.95"
  echo "validation_max_generation_length=31744"
  echo "student_model=Qwen/Qwen3-1.7B"
  echo "teacher_ref_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  echo "teacher_rl_model=hbx/JustRL-DeepSeek-1.5B"
  echo "train_data=Skywork/Skywork-OR1-RL-Data math split"
} > "${run_dir}/resolved_parameters.txt"

for required in \
  "${model_root}/Qwen3-1.7B/config.json" \
  "${model_root}/JustRL-DeepSeek-1.5B/config.json" \
  "${model_root}/DeepSeek-R1-Distill-Qwen-1.5B/config.json" \
  "${data_root}/train/skywork-or1-math-dapo-original.parquet"; do
  if [ ! -f "${required}" ]; then
    echo "missing input: ${required}" >&2
    exit 4
  fi
done

if ! visible_gpus=$(python3 "${repo_root}/scripts/select_idle_gpus.py" --snapshot "${run_dir}/gpu_selection.json" --batch-size 256); then
  echo "No idle NVIDIA GPU accessible; prepared ${run_dir}, training not started." >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="${visible_gpus}"
export GPUS_PER_NODE=$(awk -F, '{print NF}' <<< "${visible_gpus}")
nvidia-smi -L > "${run_dir}/gpu_inventory.txt" 2>&1
nvidia-smi topo -m > "${run_dir}/gpu_topology.txt" 2>&1 || true
df -h "${run_dir}" > "${run_dir}/disk.txt"
echo "Using ${GPUS_PER_NODE} idle GPU(s): ${CUDA_VISIBLE_DEVICES}" >&2

export MODEL_ROOT="${model_root}" DATA_ROOT="${data_root}"
export OUTPUT_ROOT="${run_dir}/checkpoints" LOG_ROOT="${run_dir}/logs"
export EXPERIMENT_NAME="${experiment_name}" PROJECT_NAME=direct-opd
export PYTHON_BIN="${python_bin}"
export MAX_RESP_LENGTH="${max_response_length}" MAX_PROMPT_LENGTH=1024
export MAX_VAL_RESP_LENGTH=31744
export MINI_BATCH_SIZE=64 N_RESPONSES=4
export TEMPERATURE=1.0 TRAIN_TOP_P=1.0 LOG_PROB_TOP_K=16
export TOP_K_STRATEGY=only_stu REWARD_WEIGHT_MODE=student_p
export OPTIM_LR=1e-6 KL_LOSS_COEF=1 ADAPTIVE_KL_LOSS_COEF=False
export TOTAL_TRAINING_STEPS="${training_steps}" SAVE_FREQ=5 TEST_FREQ="${test_freq}"
export DATA_SEED=42 VAL_BEFORE_TRAIN="${val_before_train}" VAL_N=32
export TEST_DATASET="['${repo_root}/datasets/eval/aime24.parquet','${repo_root}/datasets/eval/aime25.parquet']"
export ROLLOUT_DATA_DIR="${run_dir}/rollouts"
export TRAIN_DATASET="${data_root}/train/skywork-or1-math-dapo-original.parquet"

cd "${repo_root}"
exec bash scripts/train_justrl_qwen.sh
