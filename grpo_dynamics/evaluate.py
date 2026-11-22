"""Deterministic vLLM preflight or validation with lossless per-sample results."""

import argparse
from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from grpo_dynamics.prepare import PROTOCOL, sha256_file
from grpo_dynamics.record import atomic_json, sampling_seeds
from grpo_dynamics.verifier import score_answer


def generate(args):
    from vllm import LLM, SamplingParams

    protocol = PROTOCOL
    rows = pq.read_table(args.data).to_pylist()
    is_validation = args.mode == "validate"
    n = protocol["eval_samples"] if is_validation else protocol["rollout_n"]
    temperature = protocol["eval_temperature"] if is_validation else protocol["temperature"]
    top_p = protocol["eval_top_p"] if is_validation else protocol["top_p"]
    sampling = {"n": 1, "temperature": temperature, "top_p": top_p, "top_k": -1,
                "max_tokens": protocol["response_max"], "logprobs": 0}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer.apply_chat_template([r["prompt"] for r in rows], enable_thinking=False,
                                        add_generation_prompt=True, tokenize=True, return_dict=False)
    if any(len(x) > protocol["prompt_max"] for x in ids):
        raise ValueError("Manifest contains prompts exceeding fixed prompt length")
    engine = LLM(model=str(args.model), dtype="bfloat16", max_model_len=10240, seed=42,
                 tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=args.gpu_memory, enable_prefix_caching=True)
    seeds = sampling_seeds(0 if is_validation else 1, len(rows)*n, validation=is_validation)
    subset = list(range(args.rank, len(rows), args.world_size))
    if not subset:
        raise ValueError("Empty eval worker shard")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    writer = None
    try:
        for start in range(0, len(subset), args.batch_prompts):
            chosen = subset[start:start+args.batch_prompts]
            inputs = [{"prompt_token_ids": ids[i]} for i in chosen for _ in range(n)]
            params = [SamplingParams(seed=int(seeds[i*n+j]), **sampling) for i in chosen for j in range(n)]
            outputs = engine.generate(inputs, params, use_tqdm=False)
            records = []
            for k, output in enumerate(outputs):
                i, j = chosen[k//n], k % n
                sample = output.outputs[0]
                row = rows[i]
                gold = row["reward_model"]["ground_truth"]
                reward = score_answer(sample.text, gold)
                records.append({
                    "problem_id": row["extra_info"]["problem_id"], "sample_index": j,
                    "difficulty": float(row["extra_info"]["difficulty"]),
                    "sampling_seed": int(seeds[i*n+j]), "prompt_token_ids": ids[i],
                    "ground_truth": gold, "response_text": sample.text, "response_token_ids": list(sample.token_ids),
                    "response_length": len(sample.token_ids), "truncated": sample.finish_reason == "length",
                    "eos": bool(sample.token_ids and sample.token_ids[-1] == tokenizer.eos_token_id),
                    "finish_reason": sample.finish_reason, "reward": reward["score"],
                    "extracted_answer": reward["extracted_answer"],
                    "rollout_logp": [scores[token].logprob for token, scores in zip(sample.token_ids, sample.logprobs, strict=True)],
                })
            table = pa.Table.from_pylist(records)
            if writer is None:
                schema = table.schema.with_metadata({"sampling_config": json.dumps(sampling),
                                                    "dataset_sha256": sha256_file(args.data)})
                writer = pq.ParquetWriter(temp, schema, compression="zstd")
            writer.write_table(table)
            print(f"questions {min(start+len(chosen),len(subset))}/{len(subset)}", flush=True)
    finally:
        if writer:
            writer.close()
        engine.llm_engine.engine_core.shutdown()
    temp.replace(args.output)


def analyze(paths, output, baseline=None, mode="preflight", manifest=None):
    def grouped(paths):
        groups = {}
        for p in paths:
            for row in pq.read_table(p, columns=["problem_id","sample_index","reward","response_length","eos","truncated"]).to_pylist():
                sample = groups.setdefault(row["problem_id"], {})
                if row["sample_index"] in sample:
                    raise ValueError("Duplicate result for a question/sample pair")
                sample[row["sample_index"]] = row
        return groups

    current = grouped(paths)
    n = PROTOCOL["rollout_n"] if mode == "preflight" else PROTOCOL["eval_samples"]
    if any(set(samples) != set(range(n)) for samples in current.values()):
        raise ValueError("One or more groups lack requested samples")
    minimum = PROTOCOL["preflight_size"] if mode == "preflight" else PROTOCOL["validation_size"]
    if len(current) != minimum:
        raise ValueError(f"Expected {minimum} prompts, found {len(current)}")
    question_means = {q: np.mean([r["reward"] for r in samples.values()]) for q, samples in current.items()}
    means = np.array([question_means[q] for q in sorted(current)])
    flattened = [row for samples in current.values() for row in samples.values()]
    rng = np.random.default_rng(PROTOCOL["seed"])
    draws = rng.integers(len(means), size=(10000,len(means)))
    accuracy = float(means.mean())
    stats = {
        "groups": len(current), "sample_count": len(flattened), "accuracy": accuracy,
        "pass_at_1": accuracy, "mean_reward": accuracy,
        "all_correct_group_ratio": float((means == 1).mean()),
        "all_wrong_group_ratio": float((means == 0).mean()),
        "mixed_group_ratio": float(((means > 0) & (means < 1)).mean()),
        "eos_rate": float(np.mean([r["eos"] for r in flattened])),
        "truncation_rate": float(np.mean([r["truncated"] for r in flattened])),
        "length_mean": float(np.mean([r["response_length"] for r in flattened])),
        "bootstrap_ci95": np.quantile(means[draws].mean(1), [.025,.975]).tolist(),
        "input_sha256": {str(p): sha256_file(p) for p in paths},
    }
    if baseline:
        initial = grouped(baseline)
        if set(initial) != set(current):
            raise ValueError("Validation prompts differ from Step 0")
        prior = np.array([np.mean([r["reward"] for r in initial[q].values()]) for q in sorted(initial)])
        difference = means - prior
        stats["delta_vs_step0"] = float(difference.mean())
        stats["delta_paired_ci95"] = np.quantile(difference[draws].mean(1),[.025,.975]).tolist()
    if mode == "preflight":
        if manifest is None:
            raise ValueError("Preflight requires the data manifest")
        stats["manifest_sha256"] = sha256_file(manifest)
        stats["model_revision"] = PROTOCOL["model_revision"]
        stats["gate_passed"] = stats["mixed_group_ratio"] >= PROTOCOL["minimum_mixed_group_ratio"]
    atomic_json(output, stats)
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--mode", choices=["preflight","validate"], required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--gpu-memory", type=float, default=.7)
    p.add_argument("--batch-prompts", type=int, default=16)
    p = sub.add_parser("summarize")
    p.add_argument("--mode", choices=["preflight","validate"], required=True)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--baseline", type=Path, nargs="+")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args)
    else:
        analyze(args.inputs, args.output, args.baseline, args.mode, args.manifest)


if __name__ == "__main__":
    main()
