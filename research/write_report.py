"""Render the completed accuracy and policy shift report."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_dir
    profile = json.loads(args.profile.read_text())
    aime = pd.read_csv(root / "aime_comparison.csv")
    fixed = pd.read_csv(root / "prefix_checkpoint_summary.csv")
    lines = [
        "# Direct-OPD 复现与 Policy Shift 放大",
        "",
        "全参数 FSDP 每步使用 64 个 prompt、256 条 rollout 和全局一个 actor mini batch。"
        "E1 保留官方目标与自适应 KL。E0 使用论文 Eq. (4) 的精确 Student Top-16 Reverse KL。",
        "AIME 每题采样 32 条，temperature 0.7，top-p 0.95，生成上限 31744。"
        "主评估采用仓库配套的 boxed Prompt 和官方提取器。论文附录写明使用 DAPO Prompt。",
        "",
        f"Profiling 的 Step 2–4 中位数为 {profile['median_step_2_to_4_seconds']:.1f} 秒/步，"
        "不含完整 AIME 验证。",
        "",
        "## 准确率复现",
        "",
        "论文参考值为 AIME24 48.3% → 58.3%，AIME25 36.8% → 43.2%。"
        "下表变化相对 E1 Step 0，区间按题目做配对 bootstrap。",
        "",
        "| 实验 | Step | Benchmark | Avg@32 | 变化（pp） | 95% 区间（pp） |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in aime.itertuples():
        lines.append(
            f"| {row.experiment} | {row.step} | {row.benchmark} | {row.avg_at_32:.2%} | "
            f"{100 * row.delta_vs_baseline:+.2f} | "
            f"[{100 * row.paired_prompt_bootstrap_low:+.2f}, {100 * row.paired_prompt_bootstrap_high:+.2f}] |"
        )
    lines.extend([
        "", "## 真实 on-policy 候选与梯度", "",
        "低支持定义为 T1 概率小于 0.1，高 Shift 定义为 T1/T0 概率比至少 10。"
        "梯度为实际训练 loss 对当前 logit 的局部导数。", "",
    ])
    for arm in ("e1", "e0"):
        if set(aime[aime.experiment == arm].step) != {0, 40, 100, 200, 300}:
            raise ValueError(f"Incomplete AIME schedule for {arm}")
        rows = fixed[fixed.experiment == arm].set_index("step")
        if set(rows.index) != {0, 10, 20, 40, 60, 100, 150, 200, 300}:
            raise ValueError(f"Incomplete fixed-prefix schedule for {arm}")
        directory = root / arm / "on_policy"
        stats = json.loads((directory / "summary.json").read_text())
        buckets = pd.read_csv(directory / "support_buckets.csv")
        share = buckets.iloc[:4].share_of_high_shift_candidates.sum()
        lines.append(
            f"- {arm} 的 {stats['candidate_rows']:,} 个记录候选中，高 Shift 候选有 {share:.2%} 处于低 T1 支持桶。"
            f"低支持高 Shift 候选占实测绝对梯度 {stats['high_shift_low_support_share_of_absolute_gradient']:.2%}。"
            f"{stats['prefix_fraction_top1_suppressed_and_lower_candidate_reinforced']:.2%} 的位置同时出现"
            " T1 Top-1 局部受抑制和低支持高 Shift 候选局部受强化。"
        )
    lines.extend([
        "", "## 固定 Prefix Bank 与后期训练", "",
        "固定 bank 使用 1024 个原始 Student 前缀，在所有 checkpoint 上使用相同候选。"
        "Student 概率高于 T1 本身不代表错误。", "",
    ])
    for arm in ("e1", "e0"):
        bank = fixed[fixed.experiment == arm].set_index("step")
        initial, final = bank.loc[0], bank.loc[300]
        online = pd.read_csv(root / arm / "on_policy_dynamics.csv").set_index("step")
        accuracy = aime[aime.experiment == arm].groupby("step").avg_at_32.mean()
        late_steps = [100, 200, 300]
        lines.append(
            f"- {arm} 低支持高 Shift 候选的固定 bank 平均概率质量由 {initial.student_mean_mass:.2%} "
            f"变为 {final.student_mean_mass:.2%}。此类候选新成为 Student Top-1 的前缀占符合条件前缀 "
            f"{final.prefix_new_top1_fraction_given_eligible:.2%}。Step 300 相对 T1 的中位 log 概率比为 "
            f"{final.student_vs_t1_median_log_ratio:.3f}，相对 Teacher Shift 的中位 log-odds 变化为 "
            f"{final.relative_log_odds_change:.3f}。"
        )
        for step in late_steps:
            train, fixed_step = online.loc[step], bank.loc[step]
            lines.append(
                f"  - Step {step} 的 on-policy reward 为 {train.mean_policy_shift_reward:.4f}，"
                f"低支持高 Shift 质量为 {train.student_mass_low_support_high_shift_per_prefix:.2%}。"
                f"固定 bank 的 reward 为 {fixed_step.teacher_reward_on_fixed_support:.4f}，"
                f"对应质量为 {fixed_step.student_mean_mass:.2%}，AIME 两年均值为 {accuracy.loc[step]:.2%}。"
            )
        pattern = (
            (online.loc[late_steps].mean_policy_shift_reward.diff().dropna() > 0).all()
            and (online.loc[late_steps].student_mass_low_support_high_shift_per_prefix.diff().dropna() > 0).all()
            and (bank.loc[late_steps].student_mean_mass.diff().dropna() > 0).all()
            and accuracy.loc[300] <= accuracy.loc[100]
        )
        verdict = "观察到" if pattern else "未观察到"
        lines.append(
            f"  - {verdict}预定义的联合过优化模式。判据为 Step 100/200/300 的 on-policy reward "
            "以及两个状态分布上的低支持高 Shift 质量均持续上升，同时 Step 300 的 AIME 两年均值未超过 Step 100。"
        )
    lines.extend([
        "", "这些变化是描述性关联。三个后期评估点和单个训练 seed 无法识别候选强化是否导致准确率变化。",
        "各实验的 on_policy 目录包含支持度热图、完整分桶和 thinking tag ID 敏感性结果。",
        "Pass@k、长度、重复率、截断率、生成熵与训练时序保存在评估和日志文件中。",
    ])
    (root / "reproduction_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
