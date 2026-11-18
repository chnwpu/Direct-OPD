#!/usr/bin/env python3
"""Freeze 12 AIME questions and existing Student prefixes before teacher probes."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TARGETS = (512, 1024, 2048, 4096)


def category(row: dict) -> str:
    if row["correct"]:
        return "long_correct" if row["response_length"] >= 4096 else "normal_correct"
    if row["response_length"] >= 4096:
        return "long_wrong"
    return "normal_wrong"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed_trajectories", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    rows = [json.loads(line) for line in args.fixed_trajectories.open()]
    if len(rows) != 64:
        parser.error(f"expected 64 fixed trajectories, found {len(rows)}")
    # Use the first sample when a question has two frozen trajectories.
    unique = {}
    for row in rows:
        unique.setdefault(row["problem_id"], row)
    rng = random.Random(args.seed)
    buckets = {name: [] for name in ("normal_correct", "normal_wrong", "long_wrong", "long_correct")}
    for row in unique.values():
        if row["response_length"] > min(TARGETS):
            buckets[category(row)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    for name in ("normal_correct", "normal_wrong", "long_wrong"):
        selected.extend(buckets[name][:4])
    if len(selected) < 12:
        selected_ids = {row["problem_id"] for row in selected}
        remaining = [
            row for bucket in buckets.values() for row in bucket
            if row["problem_id"] not in selected_ids
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: 12 - len(selected)])
    if len(selected) < 12:
        parser.error(f"only {len(selected)} distinct questions have an unfinished 512-token prefix")

    prefixes = []
    for row in selected:
        for target in TARGETS:
            if row["response_length"] <= target:
                continue
            prefixes.append({
                "problem_id": row["problem_id"],
                "source": row["source"],
                "ground_truth": row["ground_truth"],
                "trajectory_id": row["trajectory_id"],
                "baseline_correct": row["correct"],
                "baseline_response_length": row["response_length"],
                "selection_category": category(row),
                "prefix_length": target,
                "prompt_token_ids": row["prompt_token_ids"],
                "response_prefix_token_ids": row["response_token_ids"][:target],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for prefix in prefixes:
            handle.write(json.dumps(prefix, ensure_ascii=False) + "\n")
    manifest = {
        "seed": args.seed,
        "selection_rule": "first trajectory per AIME question; up to four normal-correct, normal-wrong and long-wrong questions, then seeded fill from all remaining strata",
        "target_positions": TARGETS,
        "available_per_category": {name: len(bucket) for name, bucket in buckets.items()},
        "selected_problem_ids": [row["problem_id"] for row in selected],
        "prefix_count": len(prefixes),
        "maximum_continuations": len(prefixes) * 3 * 8,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Selected {len(selected)} questions and {len(prefixes)} unfinished prefixes")


if __name__ == "__main__":
    main()
