#!/usr/bin/env python3
"""Aggregate Eq. 20 shift and teacher-support diagnostics on fixed trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BINS = ((0, 512), (512, 1024), (1024, 2048), (2048, 4096),
        (4096, 6144), (6144, 8192), (8192, 16384))
STUDENTS = ("S0", "H2", "H4", "H6")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def quantiles(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return {str(q): float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}


def summarize_bins(rows: list[dict], minimum_length: int = 0) -> list[dict]:
    selected = [row for row in rows if row["length"] >= minimum_length]
    result = []
    for start, end in BINS:
        eligible = [row for row in selected if row["length"] > start]
        if not eligible:
            result.append({"start": start, "end": end, "trajectory_count": 0, "token_count": 0})
            continue
        segments = {
            name: np.concatenate([row[name][start : min(end, row["length"])] for row in eligible])
            for name in ("g", "absolute_candidate_shift", "t0_sampled_logprob", "t1_sampled_logprob", "t0_entropy", "t1_entropy",
                         "t0_max_probability", "t1_max_probability", "t0_support_mass", "t1_support_mass",
                         "t0_overlap", "t1_overlap", "positive_candidate_weight", "negative_candidate_weight",
                         "low_support_high_shift_weight")
        }
        g = segments["g"]
        per_trajectory = [
            (float(np.mean(row["g"][start : min(end, row["length"])])), bool(row["correct"]))
            for row in eligible
        ]
        correlation = None
        if (len(per_trajectory) >= 3 and len({correct for _, correct in per_trajectory}) == 2
                and np.std([value for value, _ in per_trajectory]) > 0):
            correlation = float(np.corrcoef(
                [value for value, _ in per_trajectory],
                [int(correct) for _, correct in per_trajectory],
            )[0, 1])
        result.append({
            "start": start,
            "end": end,
            "trajectory_count": len(eligible),
            "token_count": len(g),
            "mean_g": float(np.mean(g)),
            "mean_absolute_g": float(np.mean(np.abs(g))),
            "mean_student_weighted_absolute_candidate_shift": float(np.mean(segments["absolute_candidate_shift"])),
            "g_quantiles": quantiles(g),
            "positive_g_fraction": float(np.mean(g > 0)),
            "negative_g_fraction": float(np.mean(g < 0)),
            "mean_t0_sampled_logprob": float(np.mean(segments["t0_sampled_logprob"])),
            "mean_t1_sampled_logprob": float(np.mean(segments["t1_sampled_logprob"])),
            "mean_t0_entropy": float(np.mean(segments["t0_entropy"])),
            "mean_t1_entropy": float(np.mean(segments["t1_entropy"])),
            "mean_t0_max_probability": float(np.mean(segments["t0_max_probability"])),
            "mean_t1_max_probability": float(np.mean(segments["t1_max_probability"])),
            "mean_t0_mass_on_student_top16": float(np.mean(segments["t0_support_mass"])),
            "mean_t1_mass_on_student_top16": float(np.mean(segments["t1_support_mass"])),
            "mean_t0_top16_overlap_fraction": float(np.mean(segments["t0_overlap"])),
            "mean_t1_top16_overlap_fraction": float(np.mean(segments["t1_overlap"])),
            "mean_positive_candidate_student_mass": float(np.mean(segments["positive_candidate_weight"])),
            "mean_negative_candidate_student_mass": float(np.mean(segments["negative_candidate_weight"])),
            "mean_low_support_high_shift_student_mass": float(
                np.mean(segments["low_support_high_shift_weight"])
            ),
            "g_vs_trajectory_correct_correlation": correlation,
        })
        for field in ("anchored_direct_logit_l1", "anchored_low_support_logit_l1"):
            if field in eligible[0]:
                values = [row[field][start : min(end, row["length"])] for row in eligible]
                result[-1][f"mean_{field}"] = float(np.mean(np.concatenate(values)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed_trajectories", type=Path)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--students", nargs="+", choices=STUDENTS, default=list(STUDENTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trajectories = [json.loads(line) for line in args.fixed_trajectories.open()]
    results = {"equation": "g_t = sum_{a in Student Top16} p_S(a|s_t) [log p_T1(a|s_t) - log p_T0(a|s_t)]",
               "student_probability_weights": "raw, not renormalized within Top16",
               "correctness_label": "initial Student correctness on the frozen trajectory",
               "low_support_high_shift_rule": "both teacher candidate probabilities < 1e-4 and absolute log-ratio > 2",
               "models": {}}
    baseline_shift = {}
    for student_name in args.students:
        rows = []
        for trajectory in trajectories:
            trajectory_id = trajectory["trajectory_id"]
            base = f"trajectory_{trajectory_id:04d}.npz"
            student = load(args.stats_dir / student_name / base)
            t0 = load(args.stats_dir / "T0" / base)
            t1 = load(args.stats_dir / "T1" / base)
            length = len(trajectory["response_token_ids"])
            if any(len(data["sampled_logprob"]) != length for data in (student, t0, t1)):
                raise ValueError(f"response length mismatch for trajectory {trajectory_id}")
            student_p = np.exp(student["topk_logprob"].astype(np.float64))
            t0_lp = t0[f"candidate_logprob_{student_name}"].astype(np.float64)
            t1_lp = t1[f"candidate_logprob_{student_name}"].astype(np.float64)
            delta = t1_lp - t0_lp
            g = np.sum(student_p * delta, axis=-1)
            if not np.isfinite(g).all():
                raise ValueError(f"nonfinite Eq. 20 shift for trajectory {trajectory_id}")
            if student_name == "S0":
                baseline_shift[trajectory_id] = g
            low_support_high_shift = (t0_lp < np.log(1e-4)) & (t1_lp < np.log(1e-4)) & (np.abs(delta) > 2)
            student_mass = student_p.sum(axis=-1, keepdims=True)
            # Exact direct-loss gradient at PPO ratio 1, with detached rollout weights.
            coefficients = -student_p * delta / student_mass

            def logit_l1(gradient):
                sum_gradient = gradient.sum(axis=-1, keepdims=True)
                inside = np.abs(gradient - student_p * sum_gradient).sum(axis=-1)
                outside = np.abs(sum_gradient[:, 0]) * np.maximum(1 - student_mass[:, 0], 0)
                return inside + outside

            rows.append({
                "id": trajectory_id,
                "length": length,
                "correct": trajectory["correct"],
                "g": g,
                "absolute_candidate_shift": np.sum(student_p * np.abs(delta), axis=-1),
                "t0_sampled_logprob": t0["sampled_logprob"],
                "t1_sampled_logprob": t1["sampled_logprob"],
                "t0_entropy": t0["entropy"],
                "t1_entropy": t1["entropy"],
                "t0_max_probability": t0["max_probability"],
                "t1_max_probability": t1["max_probability"],
                "t0_support_mass": np.exp(t0_lp).sum(axis=-1),
                "t1_support_mass": np.exp(t1_lp).sum(axis=-1),
                "t0_overlap": (
                    student["topk_ids"][:, :, None] == t0["topk_ids"][:, None, :]
                ).any(axis=-1).mean(axis=-1),
                "t1_overlap": (
                    student["topk_ids"][:, :, None] == t1["topk_ids"][:, None, :]
                ).any(axis=-1).mean(axis=-1),
                "positive_candidate_weight": np.sum(student_p * (delta > 0), axis=-1),
                "negative_candidate_weight": np.sum(student_p * (delta < 0), axis=-1),
                "low_support_high_shift_weight": np.sum(student_p * low_support_high_shift, axis=-1),
                "anchored_direct_logit_l1": logit_l1(coefficients),
                "anchored_low_support_logit_l1": logit_l1(coefficients * low_support_high_shift),
            })
        cumulative = []
        for horizon in (512, 1024, 2048, 4096, 6144, 8192, 16384):
            values = [float(np.sum(row["g"][:horizon])) for row in rows if row["length"] >= horizon]
            cumulative.append({
                "horizon": horizon,
                "trajectory_count": len(values),
                "mean_G": float(np.mean(values)) if values else None,
                "standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            })
        long_curves = [np.cumsum(row["g"]) for row in rows if row["length"] >= 16384]
        curve = np.mean(long_curves, axis=0) if long_curves else None
        changes = []
        if student_name != "S0" and baseline_shift:
            for start, end in BINS:
                # Compare the same positions in the same trajectories in both policies.
                differences = [
                    float((row["g"][start:end] - baseline_shift[row["id"]][start:end]).sum())
                    for row in rows if row["length"] >= end
                ]
                changes.append({
                    "start": start, "end": end, "trajectory_count": len(differences),
                    "mean_segment_shift_change": float(np.mean(differences)) if differences else None,
                    "standard_error": (
                        float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
                        if len(differences) > 1 else None
                    ),
                })
        results["models"][student_name] = {
            "trajectory_count": len(rows),
            "cumulative_shift": cumulative,
            "paired_shift_change_vs_S0": changes,
            "bins_all_available": summarize_bins(rows),
            "bins_same_long_trajectory_subset": {
                str(threshold): summarize_bins(rows, minimum_length=threshold)
                for threshold in (4096, 6144, 8192, 16384)
            },
            "cumulative_curve_same_16384_trajectory_subset": [
                {"position": position, "mean_G": float(curve[position - 1])}
                for position in range(128, 16385, 128)
            ] if curve is not None else [],
        }
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote teacher shift statistics to {args.output}")


if __name__ == "__main__":
    main()
