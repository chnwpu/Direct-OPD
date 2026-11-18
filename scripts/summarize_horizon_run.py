#!/usr/bin/env python3
"""Summarize measured training tokens, updates and GPU-hours from one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_key_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(line.rstrip("\n").split("=", 1) for line in path.open() if "=" in line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    rollouts = run_dir / "rollouts"
    token_files = sorted(rollouts.glob("step_*_token_ids.npz"))
    if not token_files:
        parser.error(f"no exact token dumps in {rollouts}")

    steps = []
    cumulative_generated = 0
    cumulative_training = 0
    cumulative_updates = 0
    missing_update_metrics = False
    for token_file in token_files:
        step = int(token_file.name.split("_")[1])
        with np.load(token_file, allow_pickle=False) as arrays:
            prompts = arrays["prompts"]
            responses = arrays["responses"]
            response_mask = arrays["response_mask"]
            attention_mask = arrays["attention_mask"]
            if int(arrays["global_step"]) != step:
                raise ValueError(f"step mismatch in {token_file}")
            if prompts.shape[0] != responses.shape[0] or responses.shape != response_mask.shape:
                raise ValueError(f"tensor shape mismatch in {token_file}")
            if attention_mask.shape != (responses.shape[0], prompts.shape[1] + responses.shape[1]):
                raise ValueError(f"attention mask shape mismatch in {token_file}")
            if not np.array_equal(attention_mask[:, prompts.shape[1] :], response_mask):
                raise ValueError(f"response and attention masks differ in {token_file}")
            generated = int(response_mask.sum(dtype=np.int64))
            training = int(attention_mask.sum(dtype=np.int64))
            samples = int(responses.shape[0])

        metrics_path = rollouts / f"step_{step:04d}_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        updates = metrics.get("actor/optimizer_updates_step")
        if updates is None:
            missing_update_metrics = True
        else:
            if abs(updates - round(updates)) > 1e-6:
                raise ValueError(f"noninteger optimizer update count in {metrics_path}: {updates}")
            updates = int(round(updates))
            cumulative_updates += updates
        cumulative_generated += generated
        cumulative_training += training
        steps.append({
            "step": step,
            "samples": samples,
            "generated_tokens": generated,
            "training_tokens": training,
            "optimizer_updates": updates,
            "cumulative_generated_tokens": cumulative_generated,
            "cumulative_training_tokens": cumulative_training,
            "cumulative_optimizer_updates": None if missing_update_metrics else cumulative_updates,
        })

    timing = parse_key_values(run_dir / "timing.txt")
    summary = {
        "run_dir": str(run_dir.resolve()),
        "code_commit": (run_dir / "code_commit.txt").read_text().strip(),
        "resolved_parameters": parse_key_values(run_dir / "resolved_parameters.txt"),
        "steps": steps,
        "totals": {
            "generated_tokens": cumulative_generated,
            "training_tokens": cumulative_training,
            "optimizer_updates": None if missing_update_metrics else cumulative_updates,
            "allocated_gpu_hours": float(timing["allocated_gpu_hours"]) if "allocated_gpu_hours" in timing else None,
            "wall_seconds": int(timing["wall_seconds"]) if "wall_seconds" in timing else None,
        },
    }
    output = run_dir / "run_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
