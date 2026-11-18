#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
e1="${E1_RUN_DIR:-$root/runs/e1_direct_opd}"
e0="${E0_RUN_DIR:-$root/../direct-opd-e0/runs/e0_vanilla_opd}"
output="${ANALYSIS_DIR:-$root/runs/analysis}"
python="$root/.envs/direct-opd/bin/python"
student="$root/models/Qwen3-1.7B"
t0="$root/models/DeepSeek-R1-Distill-Qwen-1.5B"
t1="$root/models/JustRL-DeepSeek-1.5B"
export PYTHONPATH="$root:$root/verl"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$output/mpl_cache"
mkdir -p "$output"
baseline="$e1/checkpoints/e1_shift_amplification/outputs/validation_log/e1_shift_amplification/step0_token_trajectories.pt"
candidates="$output/prefix_candidates.pt"
bank="$output/prefix_bank.pt"

if [ ! -s "$bank" ]; then
    "$python" research/build_prefix_candidates.py --step0-tokens "$baseline" --output "$candidates"
    "$python" research/score_prefix_bank.py freeze --input "$candidates" --model "$student" --output "$bank"
fi
for teacher in t0 t1; do
    if [ ! -s "$output/$teacher.pt" ]; then
        "$python" research/score_prefix_bank.py score --input "$bank" --model "${!teacher}" --output "$output/$teacher.pt"
    fi
done

prefix_args=()
eval_args=()
for experiment in e1 e0; do
    if [ "$experiment" = e1 ]; then
        name=e1_shift_amplification
    else
        name=e0_vanilla_opd
    fi
    run="${!experiment}"
    checkpoint_root="$run/checkpoints/$name"
    output_dir="$output/$experiment"
    mkdir -p "$output_dir"
    "$python" research/rescore_teacher_rank.py --model "$t1" --diagnostic-dir "$run/diagnostics"
    "$python" research/analyze_on_policy.py --diagnostic-dir "$run/diagnostics" --output-dir "$output_dir/on_policy"
    for step in 0 10 20 40 60 100 150 200 300; do
        if [ "$step" = 0 ]; then
            model="$student"
        else
            model="$output_dir/merged_step_$step"
            if [ ! -e "$model/.merge_complete" ]; then
                "$python" -m verl.model_merger merge --backend fsdp --use_cpu_initialization \
                    --local_dir "$checkpoint_root/global_step_$step/actor" --target_dir "$model"
                touch "$model/.merge_complete"
            fi
        fi
        score="$output_dir/prefix_step_$step.pt"
        if [ ! -s "$score" ]; then
            "$python" research/score_prefix_bank.py score --input "$bank" --model "$model" --output "$score"
        fi
        prefix_args+=(--checkpoint "$experiment" "$step" "$score")
        case "$step" in
            0|40|100|200|300)
                tokens="$checkpoint_root/outputs/validation_log/$name/step${step}_token_trajectories.pt"
                entropy="$output_dir/eval_entropy_step_$step.json"
                if [ ! -s "$entropy" ]; then
                    "$python" research/score_eval_entropy.py --model "$model" --tokens "$tokens" --output "$entropy"
                fi
                eval_args+=(--checkpoint "$experiment" "$step" "$tokens")
                ;;
        esac
    done
done
"$python" research/analyze_prefix_bank.py --bank "$bank" --t0 "$output/t0.pt" --t1 "$output/t1.pt" \
    --output-dir "$output" "${prefix_args[@]}"
"$python" research/compare_evaluations.py --baseline "$baseline" --output "$output/aime_comparison.csv" "${eval_args[@]}"
"$python" research/export_training_dynamics.py --e1-metrics "$e1/logs/metrics.jsonl" \
    --e0-metrics "$e0/logs/metrics.jsonl" --analysis-dir "$output"
