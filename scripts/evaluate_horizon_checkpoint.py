#!/usr/bin/env python3
"""Evaluate an exported checkpoint with the official AIME verifier and Avg@32 settings."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.ttrl_math import compute_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Hugging Face checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count <= 60:
        parser.error("invalid shard index or count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    datasets = Path(__file__).resolve().parents[1] / "datasets" / "eval"
    rows = []
    for name in ("aime24", "aime25"):
        data = pq.read_table(datasets / f"{name}.parquet").to_pylist()
        if len(data) != 30:
            raise ValueError(f"expected 30 problems in {name}")
        rows.extend(data)

    config = {
        "model": str(args.model.resolve()), "seed": args.seed, "samples_per_problem": 32,
        "temperature": 0.7, "top_p": 0.95, "max_generation_tokens": 31744, "max_model_len": 32768,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
    }
    (args.output_dir / "parameters.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [
        tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
        for row in rows
    ]
    if any(len(prompt) > 1024 for prompt in prompts):
        raise ValueError("an AIME prompt exceeds the training prompt limit")
    llm = LLM(
        model=config["model"], dtype="bfloat16", max_model_len=32768,
        max_num_seqs=10, gpu_memory_utilization=0.85, seed=args.seed,
        enable_chunked_prefill=True, enable_prefix_caching=True,
    )
    totals = defaultdict(int)
    correct = defaultdict(int)
    problem_indices = list(range(args.shard_index, len(rows), args.shard_count))
    temporary = args.output_dir / "generations.jsonl.tmp"
    with temporary.open("w") as handle:
        for start in range(0, len(problem_indices), 4):
            batch = problem_indices[start : start + 4]
            requests = [{"prompt_token_ids": prompts[index]} for index in batch]
            sampling = [
                SamplingParams(n=32, temperature=0.7, top_p=0.95, top_k=-1,
                               max_tokens=31744, repetition_penalty=1.0, seed=args.seed + index)
                for index in batch
            ]
            outputs = llm.generate(requests, sampling)
            for index, completion in zip(batch, outputs, strict=True):
                row = rows[index]
                source = row["data_source"]
                ground_truth = row["reward_model"]["ground_truth"]
                if len(completion.outputs) != 32:
                    raise ValueError(f"expected 32 completions for request {completion.request_id}")
                for item in completion.outputs:
                    response = tokenizer.decode(item.token_ids, skip_special_tokens=True)
                    passed = bool(compute_score(response, ground_truth)["acc"])
                    totals[source] += 1
                    correct[source] += passed
                    handle.write(json.dumps({
                        "source": source, "problem_id": row["extra_info"]["index"],
                        "sample_index": item.index, "ground_truth": ground_truth,
                        "correct": passed, "generated_tokens": len(item.token_ids),
                        "finish_reason": item.finish_reason, "response": response,
                    }, ensure_ascii=False) + "\n")
            handle.flush()
    temporary.replace(args.output_dir / "generations.jsonl")
    by_source = {
        source: {"correct": correct[source], "samples": count, "avg@32": correct[source] / count}
        for source, count in totals.items()
    }
    overall = sum(correct.values()) / sum(totals.values())
    results = {"avg@32": overall, "by_source": by_source, "sampling_config": config}
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
