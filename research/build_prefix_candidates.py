"""Create deterministic Step-0 evaluation prefix candidates before entropy stratification."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch


POSITION_FRACTIONS = (0.10, 0.35, 0.65, 0.90)


def build_candidates(trajectories: list[dict], seed: int = 260705394) -> list[dict]:
    rng = np.random.default_rng(seed)
    groups = {}
    for index, row in enumerate(trajectories):
        if len(row["response_ids"]) < 4:
            continue
        key = (row["data_source"], bool(row["score"] > 0))
        groups.setdefault(key, []).append(index)
    if len(groups) != 4:
        raise ValueError(f"Expected two benchmarks with correct and incorrect trajectories; got {list(groups)}")

    candidates = []
    for (data_source, correct), indices in sorted(groups.items()):
        if len(indices) < 128:
            raise ValueError(f"Need at least 128 trajectories for {data_source}, correct={correct}; got {len(indices)}")
        selected = rng.choice(indices, size=128, replace=False)
        for trajectory_index in selected:
            row = trajectories[int(trajectory_index)]
            response = row["response_ids"]
            for fraction in POSITION_FRACTIONS:
                position = min(len(response) - 1, math.floor(fraction * len(response)))
                candidates.append({
                    "source_trajectory_index": int(trajectory_index),
                    "data_source": data_source,
                    "correct": correct,
                    "relative_position": fraction,
                    "token_position": position,
                    "prefix_ids": row["prompt_ids"] + response[:position],
                    "observed_next_token_id": response[position],
                })
    assert len(candidates) == 2048
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step0-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trajectories = torch.load(args.step0_tokens, map_location="cpu", weights_only=True)
    candidates = build_candidates(trajectories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": 260705394, "candidates": candidates}, args.output)
    print(f"Saved {len(candidates)} prefix candidates to {args.output}")


if __name__ == "__main__":
    main()
