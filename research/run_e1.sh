#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR="${RUN_DIR:-${ROOT}/runs/e1_direct_opd}"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/logs" "${RUN_DIR}/diagnostics" "${RUN_DIR}/ray"

for model_file in \
  "${ROOT}/models/Qwen3-1.7B/model-00001-of-00002.safetensors" \
  "${ROOT}/models/Qwen3-1.7B/model-00002-of-00002.safetensors" \
  "${ROOT}/models/JustRL-DeepSeek-1.5B/model.safetensors" \
  "${ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors"; do
  test -s "${model_file}" || { echo "Missing model: ${model_file}" >&2; exit 1; }
done
printf '%s  %s\n' \
  '169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5' "${ROOT}/models/Qwen3-1.7B/model-00001-of-00002.safetensors" \
  '912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9' "${ROOT}/models/Qwen3-1.7B/model-00002-of-00002.safetensors" \
  '1bedcdf243c4e2e633fa04c05617809cf6a1bbc1a07221609035f6347efecffb' "${ROOT}/models/JustRL-DeepSeek-1.5B/model.safetensors" \
  '58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945' "${ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors" \
  > "${RUN_DIR}/model_sha256.txt"
sha256sum --check "${RUN_DIR}/model_sha256.txt"
test -s "${ROOT}/datasets/train/skywork-or1-math-dapo-original.parquet"
test -x "${ROOT}/.envs/direct-opd/bin/python"

export PYTHON_BIN="${ROOT}/.envs/direct-opd/bin/python"
export PATH="${ROOT}/.envs/direct-opd/bin:${PATH}"
export PYTHONNOUSERSITE=1
export RAY_ADDRESS=local
export RAY_TMPDIR
RAY_TMPDIR=$(mktemp -d /tmp/dopd-ray.XXXXXXXX)
printf '%s\n' "${RAY_TMPDIR}" > "${RUN_DIR}/ray/runtime_path.txt"
export MANAGE_RAY=False
export ACTOR_MODEL_PATH="${ROOT}/models/Qwen3-1.7B"
export REWARD_MODEL_PATH="${ROOT}/models/JustRL-DeepSeek-1.5B"
export TEACHER_REF_MODEL_PATH="${ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B"
export TRAIN_DATASET="${ROOT}/datasets/train/skywork-or1-math-dapo-original.parquet"
export TEST_DATASET="['${ROOT}/datasets/eval/aime24.parquet','${ROOT}/datasets/eval/aime25.parquet']"
export OUTPUT_ROOT="${RUN_DIR}/checkpoints"
export LOG_ROOT="${RUN_DIR}/logs"
export LOGGER="${LOGGER:-['console','file']}"
export VERL_FILE_LOGGER_PATH="${LOG_ROOT}/metrics.jsonl"
export DIAGNOSTIC_DIR="${RUN_DIR}/diagnostics"
export PROJECT_NAME=direct-opd
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-e1_shift_amplification}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-64}"
export N_RESPONSES="${N_RESPONSES:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-2048}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export TEMPERATURE=1.0
export LOG_PROB_TOP_K=16
export OPTIM_LR=1e-6
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export VAL_N="${VAL_N:-32}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export SAVE_FREQ=0
export TEST_FREQ=0
export SAVE_STEPS="${SAVE_STEPS:-[0,10,20,40,60,100,150,200,300]}"
export TEST_STEPS="${TEST_STEPS:-[40,100,200,300]}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-10}"

git -C "${ROOT}" rev-parse HEAD > "${RUN_DIR}/code_commit.txt"
"${PYTHON_BIN}" -m pip freeze > "${RUN_DIR}/environment.freeze.txt"
sha256sum "${TRAIN_DATASET}" > "${RUN_DIR}/training_data_sha256.txt"
{
  for name in ACTOR_MODEL_PATH REWARD_MODEL_PATH TEACHER_REF_MODEL_PATH TRAIN_DATASET TEST_DATASET \
      GPUS_PER_NODE MINI_BATCH_SIZE N_RESPONSES MAX_PROMPT_LENGTH MAX_RESP_LENGTH \
      MAX_VAL_RESP_LENGTH TEMPERATURE LOG_PROB_TOP_K OPTIM_LR TOTAL_TRAINING_STEPS \
      VAL_N SAVE_STEPS TEST_STEPS DIAGNOSTIC_EVERY; do
    printf '%s=%s\n' "${name}" "${!name}"
  done
} > "${RUN_DIR}/launch_config.txt"

cd "${ROOT}"
bash scripts/train_justrl_qwen.sh
