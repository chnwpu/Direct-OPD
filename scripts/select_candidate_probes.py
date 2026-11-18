#!/usr/bin/env python3
"""Select action probes from frozen shifts and measure each candidate's teacher log-ratio."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("stats_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher0", type=Path, default=Path("models/DeepSeek-R1-Distill-Qwen-1.5B"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    rows = [json.loads(line) for line in args.trajectories.open()]
    rng = np.random.default_rng(args.seed)
    groups = defaultdict(list)
    for row in rows:
        filename = f"trajectory_{row['trajectory_id']:04d}.npz"
        with np.load(args.stats_dir / "S0" / filename) as student, \
             np.load(args.stats_dir / "T0" / filename) as teacher0, \
             np.load(args.stats_dir / "T1" / filename) as teacher1:
            for group, positions in (("early", (512, 1024)), ("late", (4096, 8192))):
                for position in positions:
                    if position >= row["response_length"]:
                        continue
                    student_ids = student["topk_ids"][position]
                    t0_logprobs = teacher0["candidate_logprob_S0"][position]
                    t1_logprobs = teacher1["candidate_logprob_S0"][position]
                    t1_top_id = int(teacher1["topk_ids"][position, 0])
                    t1_top_logprob = float(teacher1["topk_logprob"][position, 0])
                    a = int(student_ids[0])
                    if a == t1_top_id:
                        continue
                    delta = t1_logprobs - t0_logprobs
                    valid = (student_ids != a) & (student_ids != t1_top_id) & (delta > 0)
                    valid &= t1_logprobs < t1_top_logprob
                    if not valid.any():
                        continue
                    c_index = int(np.where(valid, delta, -np.inf).argmax())
                    groups[group].append({
                        "group": group, "trajectory_id": row["trajectory_id"],
                        "problem_id": row["problem_id"], "prefix_length": position,
                        "ground_truth": row["ground_truth"], "prompt_token_ids": row["prompt_token_ids"],
                        "response_prefix_token_ids": row["response_token_ids"][:position],
                        "actions": {
                            "A": {"token_id": a, "t0_logprob": float(t0_logprobs[0]), "t1_logprob": float(t1_logprobs[0])},
                            "B": {"token_id": t1_top_id, "t1_logprob": t1_top_logprob},
                            "C": {
                                "token_id": int(student_ids[c_index]),
                                "t0_logprob": float(t0_logprobs[c_index]),
                                "t1_logprob": float(t1_logprobs[c_index]),
                            },
                        },
                    })
    selected = []
    for group in ("early", "late"):
        population = groups[group]
        seen = set()
        for index in rng.permutation(len(population)):
            probe = population[index]
            if probe["problem_id"] in seen:
                continue
            selected.append(probe)
            seen.add(probe["problem_id"])
            if len(seen) == 12:
                break
    if not selected:
        parser.error("no eligible prefixes with three distinct actions")
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher0, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
    ).eval().to("cuda:0")
    with torch.inference_mode():
        for probe in selected:
            prefix = probe["prompt_token_ids"] + probe["response_prefix_token_ids"]
            ids = torch.tensor([prefix], dtype=torch.long, device="cuda:0")
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1]
            logprobs = logits.log_softmax(dim=-1)
            b = probe["actions"]["B"]
            b["t0_logprob"] = float(logprobs[b["token_id"]].item())
            for action in probe["actions"].values():
                action["direct_opd_reward"] = action["t1_logprob"] - action["t0_logprob"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(probe) + "\n" for probe in selected))
    temporary.replace(args.output)
    print(f"Selected {len(selected)} eligible probes from {len(groups['early'])} early and {len(groups['late'])} late prefixes")


if __name__ == "__main__":
    main()
