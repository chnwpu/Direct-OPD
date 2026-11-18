#!/usr/bin/env python3
"""Summarize Avg@32 and paired AIME accuracy gaps."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def problem_accuracy(path):
    scores = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if "problem_id" in row:
                problem = (row["source"], row["problem_id"])
                score = int(row["correct"])
            else:
                problem = (row["input"], str(row["gts"]))
                score = int(row["acc"])
            scores[problem].append(score)
    if len(scores) != 60 or set(map(len, scores.values())) != {32}:
        raise ValueError(f"expected 60 problems with 32 samples each in {path}")
    return {problem: float(np.mean(values)) for problem, values in scores.items()}


def paired_gap(left, right, rng):
    if set(left) != set(right):
        raise ValueError("paired evaluations have different problems")
    differences = np.array([left[key] - right[key] for key in sorted(left)])
    bootstrap = rng.choice(differences, size=(20_000, len(differences)), replace=True).mean(axis=-1)
    return {
        "left_minus_right_accuracy": float(differences.mean()),
        "paired_question_bootstrap_95_percent_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("h2", "h4", "h6", "analysis"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    formal = {}
    averages = {}
    for horizon, run_dir in (("H2", args.h2), ("H4", args.h4), ("H6", args.h6)):
        formal[horizon] = {}
        for step in (0, 20, 40):
            matches = list(run_dir.glob(f"checkpoints/*/outputs/validation_log/*/{step}.jsonl"))
            if len(matches) != 1:
                raise ValueError(f"expected one AIME validation log for {horizon} Step {step}: found {matches}")
            formal[horizon][step] = problem_accuracy(matches[0])
        averages[horizon] = {step: float(np.mean(list(scores.values()))) for step, scores in formal[horizon].items()}
    match = json.loads((args.analysis / "token_matched_h2_h6.json").read_text())
    step = match["comparison"]["step"]
    matched_h2 = problem_accuracy(args.analysis / "matched_h2_step_40/generations.jsonl")
    matched_h6 = problem_accuracy(args.analysis / f"matched_h6_step_{step}/generations.jsonl")
    results = {
        "formal_avg_at_32": averages,
        "h2_minus_h6_step_40": paired_gap(formal["H2"][40], formal["H6"][40], rng),
        "token_matched": {
            **match,
            "h2_step_40_avg_at_32": float(np.mean(list(matched_h2.values()))),
            "h6_matched_avg_at_32": float(np.mean(list(matched_h6.values()))),
            **paired_gap(matched_h2, matched_h6, rng),
        },
    }
    (args.analysis / "accuracy_comparison.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
