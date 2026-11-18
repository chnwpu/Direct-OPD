#!/usr/bin/env python3
"""Estimate teacher continuation advantage with finite-sample uncertainty."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def interval(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--s0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shift-stats-dir", type=Path, help="S0, T0, and T1 teacher-forced stats")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    draws = 20_000
    by_key = defaultdict(lambda: defaultdict(list))
    for model, path in (("T0", args.t0), ("T1", args.t1), ("S0", args.s0)):
        for line in path.open():
            row = json.loads(line)
            if row["model_name"] != model:
                raise ValueError(f"{path} contains {row['model_name']} rather than {model}")
            key = (row["problem_id"], row["trajectory_id"], row["prefix_length"])
            by_key[key][model].append(row)

    per_prefix = []
    advantage_draws = {}
    for key in sorted(by_key):
        samples = by_key[key]
        if any(len(samples[model]) != 8 for model in ("T0", "T1", "S0")):
            raise ValueError(f"incomplete or duplicated eight-sample group: {key}")
        estimates = {}
        posteriors = {}
        for model in ("T0", "T1", "S0"):
            successes = sum(bool(row["correct"]) for row in samples[model])
            posterior = rng.beta(successes + 0.5, 8 - successes + 0.5, size=draws)
            posteriors[model] = posterior
            estimates[model] = {
                "successes": successes,
                "samples": 8,
                "success_rate": successes / 8,
                "jeffreys_95_percent_interval": interval(posterior),
                "mean_additional_tokens": float(np.mean([row["generated_tokens"] for row in samples[model]])),
            }
        difference = posteriors["T1"] - posteriors["T0"]
        advantage_draws[key] = difference
        entry = {
            "problem_id": key[0],
            "trajectory_id": key[1],
            "prefix_length": key[2],
            "models": estimates,
            "teacher_advantage_F": estimates["T1"]["success_rate"] - estimates["T0"]["success_rate"],
            "teacher_advantage_95_percent_interval": interval(difference),
        }
        if args.shift_stats_dir:
            filename = f"trajectory_{key[1]:04d}.npz"
            position = key[2]  # Predict the next token after this many response tokens.
            with np.load(args.shift_stats_dir / "S0" / filename) as student, \
                 np.load(args.shift_stats_dir / "T0" / filename) as teacher0, \
                 np.load(args.shift_stats_dir / "T1" / filename) as teacher1:
                weights = np.exp(student["topk_logprob"][position].astype(np.float64))
                shift = teacher1["candidate_logprob_S0"][position] - teacher0["candidate_logprob_S0"][position]
                entry["next_token_eq20_shift"] = float(np.dot(weights, shift))
        per_prefix.append(entry)

    def summarize_length(rows: list[dict]) -> list[dict]:
        result = []
        for length in (512, 1024, 2048, 4096):
            subset = [row for row in rows if row["prefix_length"] == length]
            if not subset:
                continue
            arrays = [advantage_draws[(row["problem_id"], row["trajectory_id"], length)] for row in subset]
            entry = {
                "prefix_length": length,
                "prefix_count": len(subset),
                "mean_observed_F": float(np.mean([row["teacher_advantage_F"] for row in subset])),
                "mean_F_95_percent_interval": interval(np.mean(arrays, axis=0)),
            }
            if args.shift_stats_dir and len(subset) >= 3:
                shifts = np.array([row["next_token_eq20_shift"] for row in subset])
                centered_shift = shifts - shifts.mean()
                if np.linalg.norm(centered_shift) > 0:
                    samples = np.stack(arrays)
                    centered_samples = samples - samples.mean(axis=0)
                    correlations = centered_shift @ centered_samples / (
                        np.linalg.norm(centered_shift) * np.linalg.norm(centered_samples, axis=0)
                    )
                    entry["shift_fidelity_correlation_median"] = float(np.median(correlations))
                    entry["shift_fidelity_correlation_95_percent_interval"] = interval(correlations)
            result.append(entry)
        return result

    long_ids = {row["problem_id"] for row in per_prefix if row["prefix_length"] == 4096}
    output = {
        "uncertainty_method": "20,000 seeded Monte Carlo draws from independent Jeffreys Beta(successes+0.5, failures+0.5) posteriors; 2.5 and 97.5 percentiles",
        "uncertainty_scope": "continuation sampling uncertainty conditional on these selected prefixes; does not include uncertainty from sampling mathematical problems",
        "per_prefix": per_prefix,
        "by_length_all_available": summarize_length(per_prefix),
        "by_length_same_4096_survivors": summarize_length(
            [row for row in per_prefix if row["problem_id"] in long_ids]
        ),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Summarized {len(per_prefix)} prefixes to {args.output}")


if __name__ == "__main__":
    main()
