#!/usr/bin/env python3
"""Stream teacher-forced logits over frozen trajectories without backward passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


STUDENTS = ("S0", "H2", "H4", "H6")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name", choices=STUDENTS + ("T0", "T1"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-stats-dir", type=Path, help="Directory with S0/H2/H4/H6 stats for teacher runs")
    parser.add_argument("--students", nargs="+", choices=STUDENTS, default=list(STUDENTS))
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--limit-trajectories", type=int)
    parser.add_argument("--response-limit", type=int)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("chunk size must be positive")
    is_teacher = args.model_name in ("T0", "T1")
    if is_teacher and args.student_stats_dir is None:
        parser.error("teacher runs require --student-stats-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "model_name": args.model_name,
        "model_path": str(args.model.resolve()),
        "trajectory_sha256": sha256(args.trajectories),
        "chunk_size": args.chunk_size,
        "limit_trajectories": args.limit_trajectories,
        "response_limit": args.response_limit,
        "dtype": "float32",
        "top_k": 16,
        "student_candidate_sets": args.students if is_teacher else [],
        "teacher_conditioning": "raw initial-student token IDs",
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error(f"existing run manifest differs: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval().to("cuda:0")
    vocab_size = model.get_output_embeddings().weight.shape[0]
    torch.set_grad_enabled(False)

    with args.trajectories.open() as handle:
        for row_number, line in enumerate(handle):
            if args.limit_trajectories is not None and row_number >= args.limit_trajectories:
                break
            trajectory = json.loads(line)
            trajectory_id = trajectory["trajectory_id"]
            output = args.output_dir / f"trajectory_{trajectory_id:04d}.npz"
            if output.exists():
                continue
            prompt = trajectory["prompt_token_ids"]
            response = trajectory["response_token_ids"]
            if args.response_limit is not None:
                response = response[: args.response_limit]
            if not prompt or not response:
                raise ValueError(f"empty prompt/response in trajectory {trajectory_id}")
            full_ids = prompt + response
            if min(full_ids) < 0 or max(full_ids) >= vocab_size:
                raise ValueError(f"trajectory {trajectory_id} contains IDs outside model vocabulary")
            prompt_length = len(prompt)
            response_length = len(response)
            inputs = torch.tensor(full_ids[:-1], dtype=torch.long, device="cuda:0").unsqueeze(0)
            candidate_ids = {}
            if is_teacher:
                for student_name in args.students:
                    path = args.student_stats_dir / student_name / f"trajectory_{trajectory_id:04d}.npz"
                    with np.load(path, allow_pickle=False) as arrays:
                        ids = arrays["topk_ids"][:response_length].copy()
                    if ids.shape != (response_length, 16):
                        raise ValueError(f"student top-k shape mismatch: {path}")
                    candidate_ids[student_name] = ids

            sampled_logprob = np.empty(response_length, dtype=np.float32)
            entropy = np.empty(response_length, dtype=np.float32)
            max_probability = np.empty(response_length, dtype=np.float32)
            topk_ids = np.empty((response_length, 16), dtype=np.int32)
            topk_logprob = np.empty((response_length, 16), dtype=np.float32)
            candidate_logprob = {
                name: np.empty((response_length, 16), dtype=np.float32)
                for name in candidate_ids
            }
            past = None
            for start in range(0, inputs.shape[1], args.chunk_size):
                end = min(start + args.chunk_size, inputs.shape[1])
                result = model(input_ids=inputs[:, start:end], past_key_values=past, use_cache=True)
                past = result.past_key_values
                first_prediction = max(start, prompt_length - 1)
                if first_prediction >= end:
                    del result
                    continue
                logits = result.logits[0, first_prediction - start :].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                response_start = first_prediction - (prompt_length - 1)
                response_end = end - (prompt_length - 1)
                next_ids = torch.tensor(
                    full_ids[first_prediction + 1 : end + 1], dtype=torch.long, device="cuda:0"
                )
                sampled_logprob[response_start:response_end] = (
                    log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1).cpu().numpy()
                )
                probabilities = log_probs.exp()
                entropy[response_start:response_end] = (
                    -(probabilities * log_probs).sum(dim=-1).cpu().numpy()
                )
                top = torch.topk(log_probs, 16, dim=-1)
                topk_ids[response_start:response_end] = top.indices.cpu().numpy().astype(np.int32)
                topk_logprob[response_start:response_end] = top.values.cpu().numpy()
                max_probability[response_start:response_end] = top.values[:, 0].exp().cpu().numpy()
                for name, ids in candidate_ids.items():
                    query = torch.as_tensor(ids[response_start:response_end], device="cuda:0", dtype=torch.long)
                    candidate_logprob[name][response_start:response_end] = (
                        log_probs.gather(-1, query).cpu().numpy()
                    )
                del result, logits, log_probs, probabilities, top
            save_data = {
                "trajectory_id": np.int32(trajectory_id),
                "prompt_length": np.int32(prompt_length),
                "response_length": np.int32(response_length),
                "sampled_logprob": sampled_logprob,
                "entropy": entropy,
                "max_probability": max_probability,
                "topk_ids": topk_ids,
                "topk_logprob": topk_logprob,
            }
            save_data.update({f"candidate_logprob_{name}": values for name, values in candidate_logprob.items()})
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **save_data)
            temporary.replace(output)
            print(f"{args.model_name} trajectory {trajectory_id}: {response_length} response tokens", flush=True)


if __name__ == "__main__":
    main()
