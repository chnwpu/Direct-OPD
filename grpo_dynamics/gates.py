"""Validate dataset bytes and empirical run gates."""

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from grpo_dynamics.prepare import PROTOCOL, PROTOCOL_PATH, sha256_file
from grpo_dynamics.record import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def verify_dataset(data):
    path = data / "manifest.json"
    manifest = read(path)
    if manifest["protocol_sha256"] != sha256_file(PROTOCOL_PATH) or manifest["revision"] != PROTOCOL["dataset_revision"]:
        raise ValueError("Dataset manifest does not match the canonical protocol")
    ids = {}
    for name, spec in manifest["splits"].items():
        p = data / f"{name}.ids"
        if sha256_file(p) != spec["ids_sha256"] or sha256_file(data / f"{name}.parquet") != spec["parquet_sha256"]:
            raise ValueError(f"Changed bytes: {name}")
        ids[name] = set(p.read_text().splitlines())
        if len(ids[name]) != spec["rows"]:
            raise ValueError(f"Duplicate problem ID: {name}")
    occupied = set()
    for name in ("train", "validation", "probe", "test"):
        if occupied & ids[name]:
            raise ValueError(f"Split overlap: {name}")
        occupied.update(ids[name])
    if not ids["preflight"] <= ids["train"]:
        raise ValueError("Preflight should be a subset of the candidate training pool")
    return sha256_file(path)


def verify_run(path, kind, digest):
    run = Path(path)
    launch = read(run / "launch.json")
    if launch["mode"] != kind or launch["data_manifest_sha256"] != digest:
        raise ValueError(f"Incompatible {kind} run: {run}")
    expected = 1 if kind == "smoke" else 2
    limit = 64 if kind == "smoke" else PROTOCOL["prompt_batch"] * PROTOCOL["rollout_n"]
    steps = []
    for step in range(1, expected + 1):
        row = read(run / "steps" / f"step_{step:03}.json")
        if row["global_step"] != step or row["generated_rollouts"] != limit:
            raise ValueError("Wrong rollout count or optimizer update index")
        if pq.ParquetFile(run / "rollouts" / f"step_{step:03}.parquet").metadata.num_rows != limit:
            raise ValueError("Incomplete rollout capture")
        if row["rollout_actor_logp_abs_mean_difference"] > 0.15:
            raise ValueError("Rollout/Actor log probabilities do not align within 0.15 nats mean absolute difference")
        if kind == "profile" and row["truncation_rate"] > 0.20:
            raise ValueError("Preflight profile exceeds 20% response truncation")
        steps.append(row)
    if not any(x["gradient_norm"] > 0 and x["effective_group_ratio"] > 0 for x in steps):
        raise ValueError(f"No nonzero gradient from mixed GRPO groups in {kind}")
    if kind == "smoke":
        artifact = run / "exports" / "step_001" / "checkpoint_manifest.json"
        exported = read(artifact)
        if not exported["inference_verified"] or exported["changed_weight_tensor_count"] < 1:
            raise ValueError("Smoke model lacks verified weight updates or independent inference")
        resumed = read(run / "resume-smoke-result.json")
        if resumed["reached_step"] != 2 or not resumed["loaded_optimizer_scheduler_rng_dataloader"]:
            raise ValueError("Smoke resume gate did not reach the second update")
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "record", "audit"])
    parser.add_argument("--mode", choices=["smoke", "profile", "train", "resume"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    digest = verify_dataset(args.data)
    if args.command == "verify":
        if args.mode in ("profile", "train", "resume"):
            preflight = read(args.data / "preflight_summary.json")
            if (not preflight["gate_passed"] or preflight["groups"] != PROTOCOL["preflight_size"]
                or preflight["manifest_sha256"] != digest or preflight["model_revision"] != PROTOCOL["model_revision"]):
                raise ValueError("Preflight gate is incomplete or incompatible")
            verify_run(os.environ["GRPO_SMOKE_RUN"], "smoke", digest)
        if args.mode in ("train", "resume"):
            verify_run(os.environ["GRPO_PROFILE_RUN"], "profile", digest)
    elif args.command == "record":
        files = dict(
            (p.name, sha256_file(p)) for p in sorted(args.model.iterdir())
            if p.is_file() and p.suffix in {".safetensors", ".json", ".txt"}
        )
        if not any(x.endswith(".safetensors") for x in files) or "tokenizer.json" not in files:
            raise ValueError("Model folder lacks HF weights or tokenizer")
        if files.get("model-00001-of-00002.safetensors") != "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5" or files.get("model-00002-of-00002.safetensors") != "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9":
            raise ValueError("1.7B weights are not the locked upstream revision")
        path = args.run / ("resume_launch.json" if args.mode == "resume" else "launch.json")
        atomic_json(path, {"mode": args.mode, "seed": PROTOCOL["seed"], "model": PROTOCOL["student"],
                          "model_revision": PROTOCOL["model_revision"], "model_files_sha256": files,
                          "data_manifest_sha256": digest, "protocol_sha256": sha256_file(PROTOCOL_PATH)})
    else:
        entries = verify_run(args.run, args.mode, digest)
        total = sum(x["total_step_time"] for x in entries)
        result = {"steps": len(entries), "total_seconds": total,
                  "tokens_per_second": sum(x["generated_tokens"] for x in entries)/total,
                  "estimated_100_update_seconds": 100*total/len(entries),
                  "max_peak_gpu_memory_gib": max(x["peak_gpu_memory_gib"] for x in entries)}
        atomic_json(args.run / "gate_summary.json", result)
        print(result)


if __name__ == "__main__":
    main()
