#!/usr/bin/env python3
"""Audit how frozen Qwen3 token IDs are interpreted by the DeepSeek teachers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


BINS = ((0, 512), (512, 1024), (1024, 2048), (2048, 4096),
        (4096, 6144), (6144, 8192), (8192, 16384))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed_trajectories", type=Path)
    parser.add_argument("--student", type=Path, default=Path("models/Qwen3-1.7B"))
    parser.add_argument("--t0", type=Path, default=Path("models/DeepSeek-R1-Distill-Qwen-1.5B"))
    parser.add_argument("--t1", type=Path, default=Path("models/JustRL-DeepSeek-1.5B"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.fixed_trajectories.open()]
    student = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    report = {"trajectory_count": len(rows), "teachers": {}}
    for teacher_name, teacher_path in (("T0", args.t0), ("T1", args.t1)):
        teacher = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
        teacher_vocab_size = AutoConfig.from_pretrained(teacher_path, local_files_only=True).vocab_size
        counts = Counter()
        prompt_counts = Counter()
        bin_counts = {f"{start}-{end}": Counter() for start, end in BINS}
        for row in rows:
            prompt_counts.update(row["prompt_token_ids"])
            for start, end in BINS:
                bin_counts[f"{start}-{end}"].update(row["response_token_ids"][start:end])
            counts.update(row["prompt_token_ids"])
            counts.update(row["response_token_ids"])
        mismatched = {
            token_id for token_id in counts
            if student.convert_ids_to_tokens(token_id) != teacher.convert_ids_to_tokens(token_id)
        }
        report["teachers"][teacher_name] = {
            "model": str(teacher_path),
            "teacher_embedding_vocab_size": teacher_vocab_size,
            "all_tokens": sum(counts.values()),
            "mismatched_token_occurrences": sum(counts[token_id] for token_id in mismatched),
            "outside_teacher_embedding_occurrences": sum(
                count for token_id, count in counts.items() if token_id >= teacher_vocab_size
            ),
            "prompt_mismatch_rate": (
                sum(prompt_counts[token_id] for token_id in mismatched) / sum(prompt_counts.values())
                if prompt_counts else None
            ),
            "response_bins": {
                name: {
                    "tokens": sum(bucket.values()),
                    "mismatch_rate": (
                        sum(bucket[token_id] for token_id in mismatched) / sum(bucket.values())
                        if bucket else None
                    ),
                }
                for name, bucket in bin_counts.items()
            },
            "mismatched_ids": [
                {
                    "id": token_id,
                    "count": counts[token_id],
                    "student_token": student.convert_ids_to_tokens(token_id),
                    "teacher_token": teacher.convert_ids_to_tokens(token_id),
                }
                for token_id in sorted(mismatched, key=lambda index: (-counts[index], index))
            ],
        }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote tokenizer alignment audit to {args.output}")


if __name__ == "__main__":
    main()
