#!/usr/bin/env python3
"""Choose the nearest saved checkpoint by measured token count, before scoring it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    summary = json.loads(path.read_text())
    if not summary.get("steps"):
        raise ValueError(f"no completed steps in {path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_summary", type=Path)
    parser.add_argument("comparison_summary", type=Path)
    parser.add_argument("--reference-step", type=int, default=40)
    parser.add_argument("--metric", choices=("generated", "training"), default="generated")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = load(args.reference_summary)
    comparison = load(args.comparison_summary)
    key = f"cumulative_{args.metric}_tokens"
    reference_step = next(
        (row for row in reference["steps"] if row["step"] == args.reference_step), None
    )
    if reference_step is None:
        parser.error(f"reference step {args.reference_step} is missing")
    target = reference_step[key]
    comparison_dir = Path(comparison["run_dir"])
    candidates = []
    checkpoint_roots = list((comparison_dir / "checkpoints").glob("*/global_step_*"))
    for row in comparison["steps"]:
        # The actual checkpoint root includes the unique experiment name.
        matches = [root for root in checkpoint_roots if root.name == f"global_step_{row['step']}"]
        if matches:
            candidates.append((row, matches[0]))
    if not candidates:
        parser.error(f"no saved comparison checkpoints found in {comparison_dir}")
    best, checkpoint_path = min(
        candidates,
        key=lambda pair: (abs(pair[0][key] - target), pair[0]["step"]),
    )
    result = {
        "selection_rule": "nearest absolute cumulative token difference among existing checkpoints",
        "metric": args.metric,
        "reference": {
            "run_dir": reference["run_dir"],
            "step": args.reference_step,
            "tokens": target,
        },
        "comparison": {
            "run_dir": comparison["run_dir"],
            "step": best["step"],
            "tokens": best[key],
            "checkpoint": str(checkpoint_path.resolve()),
        },
        "absolute_token_difference": abs(best[key] - target),
        "relative_token_difference": abs(best[key] - target) / target if target else None,
        "all_candidate_steps": [row["step"] for row, _ in candidates],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Selected step {best['step']} at {checkpoint_path}")


if __name__ == "__main__":
    main()
