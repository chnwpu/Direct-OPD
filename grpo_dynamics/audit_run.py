"""Verify completed run artifacts and write the checkpoint inventory."""

import argparse
import concurrent.futures
import json
from pathlib import Path
from statistics import median

import pyarrow.parquet as pq

from grpo_dynamics.prepare import PROTOCOL, sha256_file
from grpo_dynamics.record import atomic_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    launch = read(run / "launch.json")
    if launch["mode"] != "train":
        raise ValueError("Expected a formal training run")
    manifest = run / "data_manifest.json"
    if sha256_file(manifest) != launch["data_manifest_sha256"]:
        raise ValueError("Manifest checksum differs from launch")
    config = sha256_file(run / "logs/resolved_config-train.yaml")
    steps = []
    cumulative = 0
    for step in range(1, PROTOCOL["updates"] + 1):
        tag = f"step_{step:03d}"
        record = read(run / "steps" / f"{tag}.json")
        parquet = pq.ParquetFile(run / "rollouts" / f"{tag}.parquet")
        expected = PROTOCOL["prompt_batch"] * PROTOCOL["rollout_n"]
        cumulative += record["generated_tokens"]
        if (record["global_step"], record["prompt_count"], record["generated_rollouts"],
            parquet.metadata.num_rows, record["cumulative_generated_tokens"]) != (
                step, PROTOCOL["prompt_batch"], expected, expected, cumulative):
            raise ValueError(f"Incomplete or inconsistent records for {tag}")
        steps.append(record)

    artifacts = []
    hashes = []
    for step in range(0, 101, 10):
        tag = f"step_{step:03d}"
        export = run / "exports" / tag
        verified = read(export / "checkpoint_manifest.json")
        validation = run / "validation" / tag
        scored = read(validation / "summary.json")
        if (verified["step"] != step or not verified["inference_verified"]
            or verified["data_manifest_sha256"] != launch["data_manifest_sha256"]
            or verified["config_sha256"] != config or verified["base_revision"] != launch["model_revision"]):
            raise ValueError(f"Invalid export provenance for {tag}")
        shards = sorted(validation.glob("rank_*.parquet"))
        if (len(shards) != 8 or scored["groups"] != PROTOCOL["validation_size"]
            or scored["sample_count"] != PROTOCOL["validation_size"] * PROTOCOL["eval_samples"]
            or sum(pq.ParquetFile(x).metadata.num_rows for x in shards) != scored["sample_count"]):
            raise ValueError(f"Incomplete validation for {tag}")
        hashes.extend((export / name, digest) for name, digest in verified["files_sha256"].items())
        if len(scored["input_sha256"]) != len(shards):
            raise ValueError(f"Missing validation checksums for {tag}")
        validation_hashes = {Path(path).name: digest for path, digest in scored["input_sha256"].items()}
        hashes.extend((path, validation_hashes[path.name]) for path in shards)
        artifacts.append({
            "step": step, "export": str(export), "validation": str(validation),
            "accuracy": scored["accuracy"], "delta_vs_step0": scored.get("delta_vs_step0", 0.0),
            "delta_paired_ci95": scored.get("delta_paired_ci95"), "verified_inference": True,
        })
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for (path, expected), actual in zip(hashes, pool.map(sha256_file, [p for p, _ in hashes]), strict=True):
            if actual != expected:
                raise ValueError(f"Checksum mismatch: {path}")

    for step in (50, 100):
        folder = run / "checkpoints" / f"global_step_{step}"
        for name in ("data.pt", "driver_rng.pt"):
            if not (folder / name).stat().st_size:
                raise ValueError(f"Missing resume state {name} for step {step}")
        for rank in range(8):
            for stem in ("model", "optim", "extra_state"):
                if not (folder / "actor" / f"{stem}_world_size_8_rank_{rank}.pt").stat().st_size:
                    raise ValueError(f"Missing resume state {stem} rank {rank} for step {step}")

    audit = {
        "passed": True, "updates": len(steps), "train_rollouts": sum(x["generated_rollouts"] for x in steps),
        "train_generated_tokens": cumulative, "train_prompts": steps[-1]["cumulative_prompt_count"],
        "median_step_seconds": median(x["total_step_time"] for x in steps),
        "update_wall_seconds": sum(x["total_step_time"] for x in steps),
        "max_peak_gpu_memory_gib": max(x["peak_gpu_memory_gib"] for x in steps),
        "validation_rollouts": len(artifacts) * PROTOCOL["validation_size"] * PROTOCOL["eval_samples"],
        "checkpoint_and_validation_hashes_checked": len(hashes),
        "training_commit": (run / "git_commit.txt").read_text().strip(),
        "checkpoints": artifacts, "full_resume_state_inventory_verified_at": [50, 100],
    }
    atomic_json(run / "completion_audit.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
