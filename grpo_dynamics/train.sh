#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
source "${GRPO_MACHINE_CONFIG:-$root/machine/local.env}"
mode="${1:?usage: train.sh config|smoke|profile|train|resume RUN_ID}"
run_id="${2:?run ID required}"
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
run="$root/runs/$run_id"
protocol="$root/grpo_dynamics/config/protocol.json"
manifest="$GRPO_DATA/manifest.json"
prompt_batch=128
responses=8
max_response=8192
train_file="$GRPO_DATA/train.parquet"
updates=100
saves='[10,20,30,40,50,60,70,80,90,100]'
resume=disable
case "$mode" in
    config) ;;
    smoke) prompt_batch=8; updates=1; saves='[1]'; train_file="$GRPO_DATA/preflight.parquet" ;;
    smoke-resume) prompt_batch=8; updates=2; saves='[2]'; resume=auto; train_file="$GRPO_DATA/preflight.parquet" ;;
    profile) updates=2; saves='[2]' ;;
    train) ;;
    resume) updates=200; saves='[110,120,130,140,150,160,170,180,190,200]'; resume=auto ;;
    *) exit 2 ;;
esac

if [[ "$mode" != config ]]; then
    [[ -s "$manifest" ]] || { echo "Missing completed candidate manifest" >&2; exit 2; }
    IFS=, read -ra cards <<< "$GRPO_VISIBLE_GPUS"
    [[ "${#cards[@]}" == 8 && $(printf '%s\n' "${cards[@]}" | sort -u | wc -l) == 8 ]] || {
        echo "This protocol requires eight unique local GPU IDs" >&2; exit 2;
    }
    if [[ "$mode" != resume && "$mode" != smoke-resume ]]; then
        [[ ! -e "$run/launch.json" ]] || { echo "Run ID already exists" >&2; exit 2; }
    else
        target=100
        if [[ "$mode" == smoke-resume ]]; then target=1; fi
        printf -v formatted '%03d' "$target"
        [[ "$(cat "$run/checkpoints/latest_checkpointed_iteration.txt")" == "$target" && -s "$run/steps/step_$formatted.json" ]] ||
            { echo "Completed optimizer step $target required for resume" >&2; exit 2; }
        [[ "$(cat "$run/git_commit.txt")" == "$(git rev-parse HEAD)" ]] ||
            { echo "Resume requires the same commit" >&2; exit 2; }
        cmp "$manifest" "$run/data_manifest.json"
    fi
    gate_mode="$mode"
    if [[ "$mode" == smoke-resume ]]; then gate_mode=smoke; fi
    "${GRPO_PYTHON}" -m grpo_dynamics.gates verify --mode "$gate_mode" --data "$GRPO_DATA" --model "$GRPO_MODEL"
    for gpu in "${cards[@]}"; do
        [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]] ||
            { echo "GPU $gpu has an active compute process" >&2; exit 2; }
    done
fi

mkdir -p "$run"/{logs,checkpoints}
exec 9>"$run/.lock"
flock -n 9
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root:$root/verl"
export PYTHONHASHSEED=42
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GRPO_VISIBLE_GPUS"
export RAY_ADDRESS=local
export RAY_TMPDIR
RAY_TMPDIR=$(mktemp -d /tmp/grpod.XXXXXXXX)
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VERL_FILE_LOGGER_PATH="$run/logs/metrics.jsonl"
export HYDRA_FULL_ERROR=1
extra=()
if [[ "$mode" == config ]]; then
    extra=(--cfg job --resolve)
else
    cp "$manifest" "$run/data_manifest.json"
    cp "$protocol" "$run/protocol.json"
    git rev-parse HEAD > "$run/git_commit.txt"
    "$GRPO_PYTHON" -m pip freeze > "$run/environment.freeze.txt"
    nvidia-smi -q > "$run/nvidia-smi.txt"
    if [[ "$mode" != smoke-resume ]]; then
        "$GRPO_PYTHON" -m grpo_dynamics.gates record --mode "$mode" --data "$GRPO_DATA" --model "$GRPO_MODEL" --run "$run"
    fi
    export GRPO_CONFIG_SNAPSHOT="$run/logs/resolved_config-$mode.yaml"
fi

"$GRPO_PYTHON" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0 \
    data.seed=42 data.shuffle=True data.dataloader_num_workers=0 \
    data.train_files="$train_file" data.val_files="$GRPO_DATA/validation.parquet" \
    data.train_batch_size="$prompt_batch" data.max_prompt_length=2048 data.max_response_length="$max_response" \
    data.truncation=error data.filter_overlong_prompts=False \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$GRPO_MODEL" actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$prompt_batch" actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$GRPO_ACTOR_TOKEN_BUDGET" \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n="$responses" actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$GRPO_LOGP_TOKEN_BUDGET" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.rollout.log_prob_top_k=0 \
    actor_rollout_ref.rollout.max_model_len=10240 actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GRPO_GPU_MEMORY" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    critic.enable=False reward_model.enable=False reward_model.reward_manager=naive \
    custom_reward_function.path="$root/grpo_dynamics/verifier.py" custom_reward_function.name=compute_score \
    trainer.nnodes=1 trainer.n_gpus_per_node=8 trainer.total_epochs=2 trainer.total_training_steps="$updates" \
    trainer.project_name=grpo-dynamics trainer.experiment_name="$run_id" \
    "trainer.logger=['console','file']" trainer.is_plot=False \
    trainer.default_local_dir="$run/checkpoints" trainer.resume_mode="$resume" \
    trainer.val_before_train=False trainer.test_freq=0 trainer.save_freq=0 \
    "+trainer.save_steps=$saves" "+trainer.grpo_run_dir=$run" \
    "${extra[@]}" 2>&1 | tee -a "$run/logs/trainer-$mode.log"
