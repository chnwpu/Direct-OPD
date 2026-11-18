#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export RUN_DIR="${ROOT}/runs/e1_profile5"
export EXPERIMENT_NAME=e1_profile5
export TOTAL_TRAINING_STEPS=5
export VAL_BEFORE_TRAIN=False
export VAL_N=1
export SAVE_STEPS='[0,5]'
export TEST_STEPS='[]'

bash "${ROOT}/research/run_e1.sh"
