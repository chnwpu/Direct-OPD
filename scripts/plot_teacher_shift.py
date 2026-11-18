#!/usr/bin/env python3
"""Plot the cumulative Eq. 20 shift on the common 16K cohort."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shift_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.shift_report.read_text())["models"]
    figure, (raw_ax, change_ax) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 6.2), layout="constrained")
    colors = {"S0": "#444444", "H2": "#0072B2", "H4": "#009E73", "H6": "#D55E00"}
    baseline = None
    for name in ("S0", "H2", "H4", "H6"):
        if name not in report:
            continue
        curve = report[name]["cumulative_curve_same_16384_trajectory_subset"]
        if not curve:
            parser.error("the report has no 16384-token trajectories")
        positions = np.array([0] + [point["position"] for point in curve])
        shifts = np.array([0.0] + [point["mean_G"] for point in curve])
        raw_ax.plot(positions, shifts, label=name, color=colors[name], linewidth=1.8)
        if name == "S0":
            baseline = shifts
        elif baseline is not None:
            change_ax.plot(positions, shifts - baseline, label=name, color=colors[name], linewidth=1.8)

    sample_count = next(iter(report.values()))["cumulative_shift"][-1]["trajectory_count"]
    raw_ax.set_title(f"Fixed Student trajectories at 16,384 tokens (n={sample_count})")
    raw_ax.set_ylabel(r"Mean cumulative shift $G_T$")
    change_ax.set_ylabel(r"Change in $G_T$ from S0")
    change_ax.set_xlabel("Response tokens scored")
    for panel in (raw_ax, change_ax):
        for end in (2048, 4096, 6144):
            panel.axvline(end, color="#aaaaaa", linewidth=0.7, linestyle=":")
        panel.grid(axis="y", alpha=0.2)
        panel.legend(frameon=False)
    figure.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
