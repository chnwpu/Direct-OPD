#!/usr/bin/env python3
"""Continue frozen Student prefixes with one frozen model, eight samples each."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.ttrl_math import compute_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixes", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name", choices=("T0", "T1", "S0"), required=True)
    parser.add_argument("--student-tokenizer", type=Path, default=Path("models/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index or count")
    prefixes = [json.loads(line) for line in args.prefixes.open()]
    if len(prefixes) > 48:
        parser.error(f"expected at most 48 prefixes, found {len(prefixes)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer, local_files_only=True)
    vocab_size = AutoConfig.from_pretrained(args.model, local_files_only=True).vocab_size
    llm = LLM(
        model=str(args.model.resolve()),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=1024 + 31744,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        max_num_seqs=8,
        seed=args.seed,
    )
    completed = set()
    if args.output.exists():
        for line in args.output.open():
            row = json.loads(line)
            if row["model_name"] != args.model_name:
                parser.error(f"existing output mixes models: {args.output}")
            completed.add((row["trajectory_id"], row["prefix_length"], row["sample_index"]))
    with args.output.open("a") as handle:
        for prefix_index, prefix in enumerate(prefixes):
            if prefix_index % args.shard_count != args.shard_index:
                continue
            key = (prefix["trajectory_id"], prefix["prefix_length"])
            existing = {sample for trajectory, length, sample in completed if (trajectory, length) == key}
            if existing == set(range(8)):
                continue
            if existing:
                raise ValueError(f"incomplete eight-sample group for {key}; remove or repair it before resuming")
            response_prefix = prefix["response_prefix_token_ids"]
            prompt_ids = prefix["prompt_token_ids"] + response_prefix
            if max(prompt_ids) >= vocab_size:
                raise ValueError(f"prefix has token IDs outside {args.model_name} vocabulary")
            generated = llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                SamplingParams(
                    n=8,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=31744 - prefix["prefix_length"],
                    seed=args.seed + prefix_index,
                ),
                use_tqdm=True,
            )[0]
            for sample_index, output in enumerate(generated.outputs):
                continuation_ids = list(output.token_ids)
                full_response = (
                    student_tokenizer.decode(response_prefix, skip_special_tokens=True)
                    + tokenizer.decode(continuation_ids, skip_special_tokens=True)
                )
                row = {
                    "model_name": args.model_name,
                    "problem_id": prefix["problem_id"],
                    "trajectory_id": prefix["trajectory_id"],
                    "prefix_length": prefix["prefix_length"],
                    "sample_index": sample_index,
                    "seed": args.seed + prefix_index,
                    "continuation_token_ids": continuation_ids,
                    "generated_tokens": len(continuation_ids),
                    "finish_reason": output.finish_reason,
                    "correct": bool(compute_score(full_response, prefix["ground_truth"])["acc"]),
                    "full_response_text": full_response,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"Saved {args.model_name} continuations to {args.output}")


if __name__ == "__main__":
    main()
