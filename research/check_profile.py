"""Validate the five-step throughput run and estimate the 300 update time."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    rows = [json.loads(line) for line in (run / "logs/metrics.jsonl").read_text().splitlines()]
    steps = {row["step"]: row["data"] for row in rows}
    if set(range(1, 6)) - steps.keys():
        raise SystemExit("Missing one or more profiling updates")
    for step in range(1, 6):
        data = steps[step]
        for key in ("timing_s/step", "actor/grad_norm", "delta_opd/weighted_reward_token_mean"):
            if not math.isfinite(data[key]):
                raise SystemExit(f"Non-finite {key} on step {step}")
    if not (run / "diagnostics/on_policy_step_005.pt").is_file():
        raise SystemExit("Missing sampled diagnostics for step 5")
    if not list((run / "diagnostics").glob("gradient_step_005_*.pt")):
        raise SystemExit("Missing actual logit gradient diagnostics for step 5")
    marker = run / "checkpoints/e1_profile5/latest_checkpointed_iteration.txt"
    if marker.read_text().strip() != "5":
        raise SystemExit("Step 5 checkpoint did not finish")
    timings = {step: steps[step]["timing_s/step"] for step in range(1, 6)}
    settled = median(timings[step] for step in (2, 3, 4))
    result = {
        "step_wall_seconds": timings,
        "median_step_2_to_4_seconds": settled,
        "median_step_2_to_4_tokens_per_second_per_gpu": median(
            steps[step]["perf/throughput"] for step in (2, 3, 4)
        ),
        "estimated_300_update_hours_excluding_validation_and_checkpoints": settled * 300 / 3600,
    }
    output = json.dumps(result, indent=2)
    (run / "profile_summary.json").write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
