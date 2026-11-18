"""Summarize support, shift, and actual actor-loss gradients from E1 snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SUPPORT_EDGES = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 1.000001]
SUPPORT_LABELS = ["<0.001", "0.001-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.3", ">=0.3"]
HIGH_SHIFT = float(np.log(10.0))
THINK_TOKEN_IDS = {151648, 151649, 151667, 151668}


def flatten_snapshot(path: Path) -> pd.DataFrame:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    frames = []
    for record in payload["records"]:
        ids = record["candidate_ids"].numpy()
        n_positions, k = ids.shape
        rank = record.get("teacher_t1_rank_within_topk_or_zero")
        frames.append(pd.DataFrame({
            "step": payload["step"],
            "trajectory_index": record["batch_index"],
            "token_position": np.repeat(record["token_positions"].numpy(), k),
            "candidate_id": ids.ravel(),
            "student_probability": record["student_probability_full_vocab"].numpy().ravel(),
            "teacher_t0_probability": record["t0_probability_full_vocab"].numpy().ravel(),
            "teacher_t1_probability": record["t1_probability_full_vocab"].numpy().ravel(),
            "teacher_t1_rank_within_topk_or_zero": (
                rank.numpy().ravel() if rank is not None else np.zeros(n_positions * k, dtype=np.int64)
            ),
            "policy_shift": record["policy_shift"].numpy().ravel(),
            "weighted_policy_shift": record["weighted_policy_shift"].numpy().ravel(),
            "sampled_token_flag": record["sampled_token_flag"].numpy().ravel(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def flatten_gradients(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        ids = payload["candidate_ids"].numpy()
        n_positions, k = ids.shape
        frames.append(pd.DataFrame({
            "step": payload["step"],
            "trajectory_index": np.repeat(payload["trajectory_index"].numpy(), k),
            "token_position": np.repeat(payload["token_position"].numpy(), k),
            "candidate_id": ids.ravel(),
            "actual_logit_gradient": payload["d_actual_scaled_loss_d_candidate_logit"].numpy().ravel(),
            "current_student_probability": payload["current_student_logp_full_vocab"].exp().numpy().ravel(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    frame["support_bucket"] = pd.cut(
        frame["teacher_t1_probability"], SUPPORT_EDGES, labels=SUPPORT_LABELS,
        include_lowest=True, right=False,
    )
    frame["high_positive_shift"] = frame["policy_shift"] >= HIGH_SHIFT
    frame["absolute_gradient"] = frame["actual_logit_gradient"].abs()
    n_prefixes = frame[["step", "trajectory_index", "token_position"]].drop_duplicates().shape[0]
    grouped = frame.groupby("support_bucket", observed=False)
    summary = grouped.agg(
        candidate_count=("candidate_id", "size"),
        student_probability_mass=("student_probability", "sum"),
        shift_median=("policy_shift", "median"),
        shift_p90=("policy_shift", lambda x: x.quantile(0.9)),
        high_positive_shift_count=("high_positive_shift", "sum"),
        high_positive_shift_fraction=("high_positive_shift", "mean"),
        absolute_gradient_sum=("absolute_gradient", "sum"),
        signed_gradient_sum=("actual_logit_gradient", "sum"),
    ).reset_index()
    summary["student_mass_per_prefix"] = summary["student_probability_mass"] / max(n_prefixes, 1)
    summary["student_mass_share_in_topk"] = (
        summary["student_probability_mass"] / summary["student_probability_mass"].sum()
    )
    summary["share_of_high_shift_candidates"] = (
        summary["high_positive_shift_count"] / summary["high_positive_shift_count"].sum()
    )
    summary["absolute_gradient_share"] = (
        summary["absolute_gradient_sum"] / summary["absolute_gradient_sum"].sum()
    )
    top1 = frame[frame["teacher_t1_rank_within_topk_or_zero"] == 1]
    lower = frame[(frame["teacher_t1_probability"] < 0.1) & frame["high_positive_shift"]]
    top1_present = len(top1) / max(n_prefixes, 1)
    joint = lower[lower["actual_logit_gradient"] < 0].merge(
        top1[top1["actual_logit_gradient"] > 0][["step", "trajectory_index", "token_position"]].drop_duplicates(),
        on=["step", "trajectory_index", "token_position"],
    )
    diagnostics = {
        "prefix_fraction_top1_suppressed_and_lower_candidate_reinforced": (
            joint[["step", "trajectory_index", "token_position"]].drop_duplicates().shape[0] / max(n_prefixes, 1)
        ),
        "high_shift_low_support_candidate_count": len(lower),
        "high_shift_low_support_share_of_absolute_gradient": float(
            lower["absolute_gradient"].sum() / frame["absolute_gradient"].sum()
        ),
        "prefix_positions": n_prefixes,
        "candidate_rows": len(frame),
        "high_shift_threshold_log_ratio": HIGH_SHIFT,
        "teacher_top1_in_student_topk_fraction": top1_present,
        "teacher_top1_suppressed_fraction_when_present": (
            float((top1["actual_logit_gradient"] > 0).mean()) if len(top1) else None
        ),
        "lower_support_high_shift_reinforced_fraction": (
            float((lower["actual_logit_gradient"] < 0).mean()) if len(lower) else None
        ),
        "gradient_sign_convention": "positive gradient means gradient descent lowers the candidate logit",
    }
    return summary, diagnostics


def draw_heatmaps(frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    for name in ("teacher_t0_probability", "teacher_t1_probability"):
        x = np.log10(frame[name].clip(lower=1e-12).to_numpy())
        y = frame["policy_shift"].clip(lower=-12, upper=12).to_numpy()
        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.hist2d(x, y, bins=[np.linspace(-12, 0, 100), np.linspace(-12, 12, 100)], norm=LogNorm())
        fig.colorbar(image[3], ax=ax, label="Candidate count")
        ax.set_xlabel(f"log10 {name}")
        ax.set_ylabel("Teacher policy shift (clipped to [-12, 12] for display)")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_vs_policy_shift.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshots = sorted(args.diagnostic_dir.glob("on_policy_step_*.pt"))
    gradients = sorted(args.diagnostic_dir.glob("gradient_step_*.pt"))
    if not snapshots or not gradients:
        raise SystemExit("No matching snapshot and gradient files found")
    frame = pd.concat((flatten_snapshot(path) for path in snapshots), ignore_index=True)
    grad_frame = flatten_gradients(gradients)
    keys = ["step", "trajectory_index", "token_position", "candidate_id"]
    frame = frame.merge(grad_frame, on=keys, how="left", validate="one_to_one")
    rank_frames = []
    for path in sorted(args.diagnostic_dir.glob("teacher_ranks_step_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for row in payload["records"]:
            ids = row["candidate_ids"].numpy()
            rank_frames.append(pd.DataFrame({
                "step": payload["step"],
                "trajectory_index": row["batch_index"],
                "token_position": np.repeat(row["token_positions"].numpy(), ids.shape[-1]),
                "candidate_id": ids.ravel(),
                "teacher_t1_rank_rescored": row["teacher_t1_rank"].numpy().ravel(),
            }))
    if rank_frames:
        rank_data = pd.concat(rank_frames, ignore_index=True)
        frame = frame.merge(rank_data, on=keys, how="left", validate="one_to_one")
        # Prefer ranks from the in-training Teacher forward when available.
        known_rank = frame["teacher_t1_rank_within_topk_or_zero"]
        frame["teacher_t1_rank"] = known_rank.where(known_rank > 0, frame["teacher_t1_rank_rescored"])
    if frame["actual_logit_gradient"].isna().any():
        raise RuntimeError("Some selected candidates lack actor-loss gradient records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "on_policy_candidates.parquet", index=False)
    summary, diagnostics = summarize(frame)
    summary.to_csv(args.output_dir / "support_buckets.csv", index=False)
    by_step = []
    step_summaries = []
    for step, step_frame in frame.groupby("step", sort=True):
        buckets, stats = summarize(step_frame)
        high_shift_low_support = (step_frame["teacher_t1_probability"] < 0.1) & (
            step_frame["policy_shift"] >= HIGH_SHIFT
        )
        stats["student_mass_low_support_high_shift_per_prefix"] = float(
            step_frame.loc[high_shift_low_support, "student_probability"].sum()
            / max(stats["prefix_positions"], 1)
        )
        by_step.append(buckets.assign(step=step))
        step_summaries.append({"step": int(step), **stats})
    pd.concat(by_step, ignore_index=True).to_csv(args.output_dir / "support_buckets_by_step.csv", index=False)
    pd.DataFrame(step_summaries).to_csv(args.output_dir / "summary_by_step.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(diagnostics, indent=2))
    non_think = frame[~frame["candidate_id"].isin(THINK_TOKEN_IDS)]
    if len(non_think) != len(frame):
        bucket_sensitivity, diagnostic_sensitivity = summarize(non_think)
        bucket_sensitivity.to_csv(args.output_dir / "support_buckets_without_think_ids.csv", index=False)
        (args.output_dir / "summary_without_think_ids.json").write_text(
            json.dumps(diagnostic_sensitivity, indent=2)
        )
    draw_heatmaps(frame, args.output_dir)


if __name__ == "__main__":
    main()
