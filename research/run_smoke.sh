#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export RUN_DIR="${ROOT}/runs/e1_smoke1"
export EXPERIMENT_NAME=e1_smoke1
export MINI_BATCH_SIZE=8
export N_RESPONSES=1
export MAX_RESP_LENGTH=128
export MAX_VAL_RESP_LENGTH=128
export TOTAL_TRAINING_STEPS=1
export VAL_BEFORE_TRAIN=False
export VAL_N=1
export SAVE_STEPS='[0,1]'
export TEST_STEPS='[]'
export DIAGNOSTIC_EVERY=1

bash "${ROOT}/research/run_e1.sh"
