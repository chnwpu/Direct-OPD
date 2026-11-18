#!/usr/bin/env python3
"""Generate a frozen set of 64 initial-student AIME trajectories for analysis."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.ttrl_math import compute_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    items = []
    for source in ("aime24", "aime25"):
        table = pq.read_table(f"datasets/eval/{source}.parquet")
        for row in table.to_pylist():
            items.append({
                "problem_id": row["extra_info"]["index"],
                "source": source,
                "messages": row["prompt"],
                "ground_truth": row["reward_model"]["ground_truth"],
            })
    if len(items) != 60:
        raise ValueError(f"expected 60 AIME questions, found {len(items)}")
    # Four extra independent samples bring the frozen trajectory count to 64.
    duplicates = random.Random(args.seed).sample(range(len(items)), 4)
    selected = items + [items[index].copy() for index in duplicates]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt_ids = [
        tokenizer.apply_chat_template(item["messages"], tokenize=True, add_generation_prompt=True)
        for item in selected
    ]
    if max(map(len, prompt_ids)) > 1024:
        raise ValueError("AIME prompt exceeds the training max prompt length of 1024")

    llm = LLM(
        model=str(args.model.resolve()),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=1024 + args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        max_num_seqs=8,
        seed=args.seed,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_ids],
        [
            SamplingParams(
                n=1,
                temperature=1.0,
                top_p=1.0,
                max_tokens=args.max_new_tokens,
                seed=args.seed + index,
            )
            for index in range(len(selected))
        ],
        use_tqdm=True,
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with temporary.open("w") as handle:
            for index, (item, pids, output) in enumerate(zip(selected, prompt_ids, outputs, strict=True)):
                completion = output.outputs[0]
                response_ids = list(completion.token_ids)
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                record = {
                    "trajectory_id": index,
                    "problem_id": item["problem_id"],
                    "source": item["source"],
                    "ground_truth": item["ground_truth"],
                    "prompt_token_ids": pids,
                    "response_token_ids": response_ids,
                    "response_length": len(response_ids),
                    "finish_reason": completion.finish_reason,
                    "correct": bool(compute_score(response_text, item["ground_truth"])["acc"]),
                    "response_text": response_text,
                    "seed": args.seed + index,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {len(outputs)} frozen trajectories to {args.output}")


if __name__ == "__main__":
    main()
