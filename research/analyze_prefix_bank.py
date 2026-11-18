"""Compare checkpoint movement on the frozen Step-0 support."""

import argparse
from pathlib import Path

import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--t0", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", nargs=3, metavar=("EXPERIMENT", "STEP", "SCORES"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    def load(path):
        return torch.load(path, weights_only=True, map_location="cpu")

    prefixes = load(args.bank)["prefixes"]
    t0, t1 = (load(path)["scores"] for path in (args.t0, args.t1))
    candidates = torch.tensor([row["candidate_ids"] for row in prefixes])
    initial = torch.tensor([row["student0"]["logp"] for row in prefixes], dtype=torch.float64)

    def check_scores(scores):
        if not torch.equal(candidates, torch.tensor([row["candidate_ids"] for row in scores])):
            raise ValueError("Candidate IDs differ from the fixed prefix bank")
        return torch.tensor([row["logp"] for row in scores], dtype=torch.float64)

    pre, post = check_scores(t0), check_scores(t1)
    teacher_shift = post - pre
    post_p = post.exp()
    subset = (post_p < 0.1) & (teacher_shift >= torch.tensor(10.0).log())

    def logodds(logp):
        return logp - torch.log(-torch.expm1(logp.clamp(max=-1e-15)))

    summary = []
    candidate_frames = []
    for experiment, step, path in args.checkpoint:
        scores = load(path)["scores"]
        current = check_scores(scores)
        student_shift = current - initial
        top1 = candidates.eq(torch.tensor([row["top1"] for row in scores])[:, None])
        initial_top1 = candidates.eq(torch.tensor([row["student0"]["top1"] for row in prefixes])[:, None])
        odds_gap_change = logodds(current) - logodds(initial) - (logodds(post) - logodds(pre))
        frame = pd.DataFrame({
            "prefix_index": torch.arange(len(prefixes)).repeat_interleave(candidates.shape[-1]).numpy(),
            "candidate_id": candidates.flatten().numpy(),
            "low_support_high_shift": subset.flatten().numpy(),
            "teacher_shift": teacher_shift.flatten().numpy(),
            "t0_probability": pre.exp().flatten().numpy(),
            "t1_probability": post_p.flatten().numpy(),
            "student_probability": current.exp().flatten().numpy(),
            "student_initial_probability": initial.exp().flatten().numpy(),
            "student_shift": student_shift.flatten().numpy(),
            "student_vs_t1_log_ratio": (current - post).flatten().numpy(),
            "relative_log_odds_change": odds_gap_change.flatten().numpy(),
            "student_top1": top1.flatten().numpy(),
        })
        candidate_frames.append(frame.assign(experiment=experiment, step=int(step)))
        eligible = subset.any(-1)
        summary.append({
            "experiment": experiment, "step": int(step),
            "candidate_count": int(subset.sum()),
            "student_mean_mass": float((current.exp() * subset).sum(-1).mean()),
            "student_mean_log_shift": float(student_shift[subset].mean()),
            "student_vs_t1_median_log_ratio": float((current - post)[subset].median()),
            "relative_log_odds_change": float(odds_gap_change[subset].median()),
            "prefix_top1_fraction_given_eligible": float((top1 & subset).any(-1)[eligible].float().mean()),
            "prefix_new_top1_fraction_given_eligible": float((top1 & ~initial_top1 & subset).any(-1)[eligible].float().mean()),
            "mean_entropy": sum(row["entropy"] for row in scores) / len(scores),
            "teacher_reward_on_fixed_support": float((teacher_shift * current.softmax(-1)).sum(-1).mean()),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.output_dir / "prefix_checkpoint_summary.csv", index=False)
    pd.concat(candidate_frames, ignore_index=True).to_parquet(args.output_dir / "prefix_candidates.parquet", index=False)


if __name__ == "__main__":
    main()
