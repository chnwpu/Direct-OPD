#!/usr/bin/env python3
"""Write the six research answers from completed experiment artifacts."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def percent(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} 个百分点"


def interval(values):
    return f"[{100 * values[0]:+.2f}, {100 * values[1]:+.2f}] 个百分点"


def accuracy_judgment(gap):
    low, high = gap["paired_question_bootstrap_95_percent_interval"]
    if low > 0:
        return "该对照支持 2K 优于 6K。"
    if high < 0:
        return "该对照支持 6K 优于 2K。"
    if gap["left_minus_right_accuracy"] > 0:
        return "点估计支持 2K 优于 6K，区间包含零。"
    return "点估计未复现 2K 优于 6K，区间包含零。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for tag in ("h2", "h4", "h6", "analysis"):
        parser.add_argument(f"--{tag}", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis
    accuracy = load(root / "accuracy_comparison.json")
    shift = load(root / "teacher_shift.json")["models"]
    fidelity = load(root / "prefix_fidelity.json")
    gate = load(root / "candidate_probe_gate.json")
    candidates = load(root / "candidate_values/metrics.json") if gate["triggered"] else None

    lines = [
        "# Direct-OPD 长序列监督与 Reward Fidelity",
        "",
        "三个设置分别从同一初始 Student 训练 40 Step，采用 Fixed KL=1。"
        "下表的评估为 AIME 2024/2025 共 60 题，每题 32 次采样。",
        "",
        "| 设置 | Step 0 Avg@32 | Step 20 Avg@32 | Step 40 Avg@32 | 响应 token | 训练 token | Optimizer updates | GPU 小时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("H2", "H4", "H6"):
        run_dir = getattr(args, name.lower())
        totals = load(run_dir / "run_summary.json")["totals"]
        values = [percent(accuracy["formal_avg_at_32"][name][str(step)]) for step in (0, 20, 40)]
        lines.append(
            f"| {name} | {' | '.join(values)} | {totals['generated_tokens']:,} | "
            f"{totals['training_tokens']:,} | {totals['optimizer_updates']} | {totals['allocated_gpu_hours']:.2f} |"
        )

    formal = accuracy["h2_minus_h6_step_40"]
    match = accuracy["token_matched"]
    lines += [
        "",
        "## 1. 2K 是否优于 6K",
        "",
        f"Step 40 的 H2−H6 差值为 {pp(formal['left_minus_right_accuracy'])}，"
        f"按题目配对 bootstrap 的 95% 区间为 {interval(formal['paired_question_bootstrap_95_percent_interval'])}。"
        + accuracy_judgment(formal),
        "",
        "## 2. Token-matched 对照",
        "",
        f"H2 Step 40 生成 {match['reference']['tokens']:,} token。"
        f"最近的 H6 checkpoint 是 Step {match['comparison']['step']}，"
        f"生成 {match['comparison']['tokens']:,} token，预算相差 {percent(match['relative_token_difference'])}。",
        "",
        f"按相同离线采样协议重新评估后，H2 与 H6 分别为 "
        f"{percent(match['h2_step_40_avg_at_32'])} 与 {percent(match['h6_matched_avg_at_32'])}。"
        f"差值为 {pp(match['left_minus_right_accuracy'])}，"
        f"95% 配对区间为 {interval(match['paired_question_bootstrap_95_percent_interval'])}。"
        + accuracy_judgment(match),
        "",
        "## 3. Teacher Shift 与真实续写能力",
        "",
        "[Figure 8 复现图](figure8.pdf)采用完整到达 16,384 token 的共同轨迹子集。",
        "",
        "| Student | 16K 共同轨迹数 | 平均累计 Shift | 2048–4096 相对 S0 的平均区间 Shift 变化 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("S0", "H2", "H4", "H6"):
        model = shift[name]
        cumulative = model["cumulative_shift"][-1]
        changes = [row["mean_segment_shift_change"] for row in model["paired_shift_change_vs_S0"]
                   if row["start"] == 2048 and row["end"] == 4096]
        after_2k = f"{changes[0]:+.3f}" if changes and changes[0] is not None else "—"
        lines.append(f"| {name} | {cumulative['trajectory_count']} | {cumulative['mean_G']:.3f} | {after_2k} |")
    change = fidelity["paired_change_from_512_to_4096"]
    if change:
        lines += [
            "",
            f"在 {change['prefix_pairs']} 道有 4096-token 前缀的题上，"
            f"4096 相对 512 位置的平均绝对 Shift 变化为 "
            f"{change['mean_absolute_shift_change_4096_minus_512']:+.4f}。"
            f"T1 续写成功率的平均变化为 "
            f"{pp(change['mean_posterior_change_in_T1_success_4096_minus_512'])}，"
            f"95% 区间为 {interval(change['t1_change_95_percent_interval'])}。",
        ]
    lines += ["", "## 4. T1 相对 T0 的真实优势是否衰减", ""]
    for row in fidelity["by_length_same_4096_survivors"]:
        lines.append(
            f"- {row['prefix_length']} token，{row['prefix_count']} 个前缀。"
            f"平均观测 F 为 {pp(row['mean_observed_F'])}，"
            f"95% 区间为 {interval(row['mean_F_95_percent_interval'])}。"
        )
    if change:
        lines += [
            "",
            f"同题比较的 F 变化为 {pp(change['mean_posterior_change_in_F_4096_minus_512'])}，"
            f"95% 区间为 {interval(change['change_95_percent_interval'])}。"
            "这些区间反映每组 8 次续写的采样不确定性，条件是本次所选前缀。",
        ]
    lines += ["", "## 5. 候选动作排序", ""]
    if candidates:
        for name, group in candidates["by_group"].items():
            low, high = group["conditional_95_percent_interval"]
            lines.append(
                f"- {name} 位置的平均配对排序一致率为 {percent(group['mean_agreement'])}，"
                f"95% 区间为 [{percent(low)}, {percent(high)}]，共 {group['pairs']} 对。"
            )
    else:
        lines.append("所选前缀的统计量未达到预设异常判据，候选动作实验未触发。")
    common_bins = shift["S0"]["bins_same_long_trajectory_subset"]["6144"]
    early, late = common_bins[0], common_bins[4]
    t0_support_change = late["mean_t0_mass_on_student_top16"] - early["mean_t0_mass_on_student_top16"]
    t1_support_change = late["mean_t1_mass_on_student_top16"] - early["mean_t1_mass_on_student_top16"]
    direction = (
        "两位 Teacher 的候选支持度均下降，且前缀续写检验出现了预设的 Shift 与成功率背离。"
        if t0_support_change < 0 and t1_support_change < 0 and gate["triggered"]
        else "这批数据未同时满足两位 Teacher 候选支持度下降与续写背离的预设证据条件。"
    )
    lines += [
        "",
        "## 6. 对监督可靠性解释的判断",
        "",
        f"在至少 6144 token 的共同轨迹上，4096–6144 相对 0–512 区间的 Student top-16 "
        f"概率质量变化分别为 T0 {pp(t0_support_change)}、T1 {pp(t1_support_change)}。"
        + direction,
        "",
        "这里的实验没有独立操纵监督可靠性，数据约束的是该机制的观察性预测。"
        "续写用 Student 的原始 token ID 条件化，忠实对应训练时的输入，"
        "同时保留了 Student 与 Teacher 对部分特殊 token 的解释差异。",
        "",
        "完整统计见 [训练轨迹统计 H2](h2_training_fidelity.json)、"
        "[训练轨迹统计 H4](h4_training_fidelity.json)、[训练轨迹统计 H6](h6_training_fidelity.json)、"
        "[固定轨迹统计](teacher_shift.json)、[续写结果](prefix_fidelity.json)和"
        "[配对准确率](accuracy_comparison.json)。",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
