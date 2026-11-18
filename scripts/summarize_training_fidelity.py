#!/usr/bin/env python3
"""Aggregate teacher fidelity on actual sampled training rollouts."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from aggregate_teacher_shift import summarize_bins


def verifier_correct(generation):
    if "acc" in generation:
        return bool(generation["acc"])
    if "verifier_score" in generation:
        return generation["verifier_score"] > 0
    reward = generation["true_reward_score"]
    return (sum(reward) if isinstance(reward, list) else reward) > 0


def dapo_prompt_correct(generation, official_correct):
    """Also score the final Answer: line requested by the training prompts."""
    if official_correct:
        return True
    last_line = generation["output"].rstrip().rsplit("\n", 1)[-1]
    match = re.fullmatch(r"\s*Answer\s*:\s*(.{1,256}?)\s*", last_line, re.IGNORECASE)
    if match is None:
        return False
    from verl.utils.reward_score.ttrl_math import grade

    return bool(grade(match.group(1), str(generation["gts"]), fast=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = []
    for path in sorted(args.rollout_dir.glob("step_*_teacher_fidelity.npz")):
        step = int(path.name.split("_")[1])
        with (args.rollout_dir / f"{step}.jsonl").open() as handle:
            generations = [json.loads(line) for line in handle]
        with np.load(path, allow_pickle=False) as arrays:
            lengths = arrays["length"]
            offsets = np.r_[0, lengths.cumsum()]
            if len(lengths) != len(generations):
                raise ValueError(f"mismatched rollout counts in {path}")
            fields = {name: arrays[name] for name in arrays.files if name != "length"}
            rows = []
            for index, generation in enumerate(generations):
                selection = slice(offsets[index], offsets[index + 1])
                correct = verifier_correct(generation)
                rows.append({
                    "length": int(lengths[index]),
                    "correct": correct,
                    "dapo_prompt_correct": dapo_prompt_correct(generation, correct),
                    **{name: values[selection] for name, values in fields.items()},
                })
        metrics_file = args.rollout_dir / f"step_{step:04d}_metrics.json"
        metrics = json.loads(metrics_file.read_text())
        official = summarize_bins(rows)
        prompt_correct = summarize_bins([
            {**row, "correct": row["dapo_prompt_correct"]} for row in rows
        ])
        result.append({
            "step": step,
            "all_available": official,
            "official_verifier_correct_count": sum(row["correct"] for row in rows),
            "official_or_final_answer_line_correct_count": sum(row["dapo_prompt_correct"] for row in rows),
            "g_vs_official_or_final_answer_line_correct_by_bin": [
                {key: bin_row.get(key) for key in ("start", "end", "g_vs_trajectory_correct_correlation")}
                for bin_row in prompt_correct
            ],
            "same_long_trajectory_subset": {
                str(threshold): summarize_bins(rows, minimum_length=threshold)
                for threshold in (2048, 4096, 6144)
            },
            "actual_policy_gradient_logit_l1": {
                key: value for key, value in metrics.items()
                if key.startswith("actor/fidelity_")
            },
        })
    if not result:
        parser.error(f"no training fidelity files in {args.rollout_dir}")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
