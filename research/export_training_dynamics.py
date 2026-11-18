"""Align per-step training statistics with on-policy and fixed-bank diagnostics."""

import argparse
import json
from pathlib import Path

import pandas as pd


TRAIN_METRICS = {
    "delta_opd/weighted_reward_token_mean": "mean_policy_shift_reward",
    "diagnostics/student_reference_sampled_log_ratio": "sampled_kl_to_student_reference",
    "actor/adaptive_kl_coef_after": "adaptive_kl_coefficient",
    "actor/entropy": "student_entropy",
    "actor/grad_norm": "gradient_norm",
    "diagnostics/verifier_accuracy": "verifier_accuracy",
    "response_length/mean": "response_length",
    "response_length/clip_ratio": "truncation_rate",
    "diagnostics/repetition_4gram_rate": "repetition_rate",
}


def read_metrics(log):
    rows = []
    for line in log.read_text().splitlines():
        entry = json.loads(line)
        data = entry["data"]
        row = {"step": entry["step"]}
        row.update({key: data.get(source) for source, key in TRAIN_METRICS.items()})
        if row["adaptive_kl_coefficient"] is None:
            row["adaptive_kl_coefficient"] = data.get("actor/kl_coef")
        row.update({key: value for key, value in data.items() if key.startswith("val-diagnostics/")})
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("step", keep="last").sort_values("step")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-metrics", type=Path, required=True)
    parser.add_argument("--e0-metrics", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.analysis_dir
    for experiment, metrics in (("e1", args.e1_metrics), ("e0", args.e0_metrics)):
        train = read_metrics(metrics)
        on_policy = pd.read_csv(out / experiment / "on_policy/summary_by_step.csv")
        train.merge(on_policy, how="left", on="step", validate="one_to_one").to_csv(
            out / experiment / "on_policy_dynamics.csv", index=False
        )

    fixed = pd.read_csv(out / "prefix_checkpoint_summary.csv")
    evaluation = pd.read_csv(out / "aime_comparison.csv")
    aligned = evaluation.merge(fixed, on=["experiment", "step"], validate="many_to_one")
    aligned.to_csv(out / "fixed_prefix_and_aime_dynamics.csv", index=False)
    associations = []
    for keys, rows in aligned[aligned["step"] >= 100].groupby(["experiment", "benchmark"]):
        for measure in ("student_mean_mass", "teacher_reward_on_fixed_support", "relative_log_odds_change"):
            pair = rows[[measure, "avg_at_32"]].dropna()
            associations.append({
                "experiment": keys[0], "benchmark": keys[1], "measure": measure,
                "checkpoint_count": len(pair),
                "spearman_descriptive_only": pair.rank().corr().iloc[0, 1] if len(pair) >= 3 else None,
            })
    pd.DataFrame(associations).to_csv(out / "late_checkpoint_associations.csv", index=False)


if __name__ == "__main__":
    main()
