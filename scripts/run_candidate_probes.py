#!/usr/bin/env python3
"""Estimate Q under the frozen Student for forced A, B, and C tokens."""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.ttrl_math import compute_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("--student", type=Path, default=Path("models/Qwen3-1.7B"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.probes.open()]
    if not 0 < len(rows) <= 24:
        parser.error("expected between 1 and 24 probes")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "student": str(args.student.resolve()), "seed": args.seed, "samples_per_action": 16,
        "temperature": 0.7, "top_p": 0.95, "max_total_response_tokens": 31744,
    }
    (args.output_dir / "parameters.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    stopping_ids = set(tokenizer.all_special_ids).intersection(
        {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    )
    llm = LLM(
        model=config["student"], dtype="bfloat16", max_model_len=32768,
        max_num_seqs=48, gpu_memory_utilization=0.85, seed=args.seed,
        enable_chunked_prefill=True, enable_prefix_caching=True,
    )
    rng = np.random.default_rng(args.seed)
    posterior_agreement = defaultdict(list)
    results = []
    with (args.output_dir / "generations.jsonl").open("w") as log:
        for index, probe in enumerate(rows):
            base = probe["response_prefix_token_ids"]
            actions = probe["actions"]
            if set(actions) != {"A", "B", "C"} or len({a["token_id"] for a in actions.values()}) != 3:
                raise ValueError(f"invalid action set for probe {index}")
            completed = {}
            sampled_labels = [label for label in ("A", "B", "C") if actions[label]["token_id"] not in stopping_ids]
            if sampled_labels:
                prompts = [
                    {"prompt_token_ids": probe["prompt_token_ids"] + base + [actions[label]["token_id"]]}
                    for label in sampled_labels
                ]
                sampling = [
                    SamplingParams(n=16, temperature=0.7, top_p=0.95,
                                   max_tokens=31744 - len(base) - 1, seed=args.seed + 3 * index + offset)
                    for offset in range(len(prompts))
                ]
                outputs = llm.generate(prompts, sampling)
                completed = {
                    label: [list(item.token_ids) for item in completion.outputs]
                    for label, completion in zip(sampled_labels, outputs, strict=True)
                }
            estimates = {}
            posterior = {}
            for label in ("A", "B", "C"):
                token = actions[label]["token_id"]
                continuations = completed.get(label, [[] for _ in range(16)])
                if len(continuations) != 16:
                    raise ValueError(f"expected 16 continuations for probe {index} action {label}")
                flags = []
                for sample, suffix in enumerate(continuations):
                    full_answer = tokenizer.decode(base + [token] + suffix, skip_special_tokens=True)
                    passed = bool(compute_score(full_answer, probe["ground_truth"])["acc"])
                    flags.append(passed)
                    log.write(json.dumps({
                        "probe": index, "action": label, "sample": sample, "correct": passed,
                        "generated_tokens": len(suffix), "continuation_token_ids": suffix,
                    }) + "\n")
                wins = sum(flags)
                samples = (np.full(20_000, wins / 16) if token in stopping_ids
                           else rng.beta(wins + 0.5, 16 - wins + 0.5, size=20_000))
                posterior[label] = samples
                estimates[label] = {
                    **actions[label], "successes": wins, "samples": 16, "q": wins / 16,
                    "q_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
                    "forced_stop_token": token in stopping_ids,
                }
            pairs = []
            for a, b in itertools.combinations(("A", "B", "C"), 2):
                reward_diff = estimates[a]["direct_opd_reward"] - estimates[b]["direct_opd_reward"]
                value_diff = estimates[a]["q"] - estimates[b]["q"]
                if reward_diff == 0:
                    continue
                signed_posterior = np.sign(reward_diff) * (posterior[a] - posterior[b])
                agreement = (signed_posterior > 0) + 0.5 * (signed_posterior == 0)
                posterior_agreement[probe["group"]].append(agreement)
                pairs.append({
                    "actions": [a, b], "reward_difference": reward_diff, "q_difference": value_diff,
                    "observed_agreement": float(np.sign(reward_diff * value_diff) * 0.5 + 0.5),
                    "posterior_agreement": float(agreement.mean()),
                })
            results.append({
                "probe": index, "group": probe["group"], "problem_id": probe["problem_id"],
                "prefix_length": probe["prefix_length"], "candidates": estimates, "pairs": pairs,
            })
            log.flush()
    groups = {}
    for group, arrays in posterior_agreement.items():
        draws = np.mean(arrays, axis=0)
        groups[group] = {
            "pairs": len(arrays),
            "mean_agreement": float(draws.mean()),
            "conditional_95_percent_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"per_probe": results, "by_group": groups}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
