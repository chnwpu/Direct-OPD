"""Compute AIME accuracy and paired prompt bootstrap intervals by checkpoint."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_accuracy(file):
    groups = defaultdict(lambda: defaultdict(list))
    for row in torch.load(file, map_location="cpu", weights_only=True):
        groups[row["data_source"]][tuple(row["prompt_ids"])].append(float(row["score"] > 0))
    for group in groups.values():
        if len(group) != 30 or any(len(scores) != 32 for scores in group.values()):
            raise ValueError(f"{file} does not have 30 questions with 32 samples each")
    return {source: {key: np.mean(scores) for key, scores in group.items()}
            for source, group in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", nargs=3, metavar=("EXPERIMENT", "STEP", "TOKENS"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = load_accuracy(args.baseline)
    rng = np.random.default_rng(260705394)
    draws = rng.integers(0, 30, size=(10000, 30))
    records = []
    for experiment, step, file in args.checkpoint:
        current = load_accuracy(file)
        if current.keys() != baseline.keys():
            raise ValueError(f"Benchmark names differ for {file}")
        for source, group in current.items():
            initial = baseline[source]
            if group.keys() != initial.keys():
                raise ValueError(f"Prompt token IDs differ for {file} and {source}")
            order = sorted(group)
            deltas = np.array([group[key] - initial[key] for key in order])
            interval = np.quantile(deltas[draws].mean(-1), [0.025, 0.975])
            records.append({
                "experiment": experiment, "step": int(step), "benchmark": source,
                "avg_at_32": np.mean(list(group.values())), "delta_vs_baseline": deltas.mean(),
                "paired_prompt_bootstrap_low": interval[0], "paired_prompt_bootstrap_high": interval[1],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
