"""Score a fixed held-out pilot bank across all archived Teachers."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from grpo_dynamics.dynamics import teacher
from grpo_dynamics.prepare import sha256_file
from grpo_dynamics.record import atomic_json


def freeze(run, out):
    candidates = {0: [], 1: []}
    for path in sorted((run / "validation/step_000").glob("rank_*.parquet")):
        for row in pq.read_table(path, columns=["problem_id", "sample_index", "prompt_token_ids",
                                              "response_token_ids", "reward"]).to_pylist():
            if len(row["response_token_ids"]) >= 4 and row["reward"] in (0, 1):
                candidates[int(row["reward"])].append(row)
    selected = []
    used = set()
    for reward in (0, 1):
        ordered = sorted(candidates[reward], key=lambda r: hashlib.sha256(
            f"42:pilot:{r['problem_id']}:{r['sample_index']}".encode()).digest())
        count = 0
        for row in ordered:
            if row["problem_id"] in used:
                continue
            used.add(row["problem_id"])
            response = row["response_token_ids"]
            trajectory = len(selected) // 2
            for fraction in (.25, .75):
                position = int(len(response)*fraction)
                selected.append({
                    "trajectory_id": trajectory, "problem_id": row["problem_id"], "sample_index": row["sample_index"],
                    "reward": reward, "relative_position": fraction,
                    "prefix": row["prompt_token_ids"] + response[:position], "observed_token": response[position],
                })
            count += 1
            if count == 16:
                break
        if count != 16:
            raise ValueError("Need 16 distinct held-out problems per reward label")
    atomic_json(out / "bank.json", {"source": "step0_fixed_validation", "rows": selected})


def score(path, out, step):
    import torch
    from transformers import AutoModelForCausalLM
    bank = out / "bank.json"
    rows = json.loads(bank.read_text())["rows"]
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    result = np.empty((len(rows), model.config.vocab_size), dtype=np.float32)
    ordered = sorted(range(len(rows)), key=lambda i: len(rows[i]["prefix"]))
    with torch.inference_mode():
        for offset in range(0, len(rows), 4):
            indexes = ordered[offset:offset+4]
            prefixes = [rows[i]["prefix"] for i in indexes]
            width = len(prefixes[-1])
            padded = torch.tensor([[0]*(width-len(ids))+ids for ids in prefixes], device="cuda")
            mask = torch.tensor([[0]*(width-len(ids))+[1]*len(ids) for ids in prefixes], device="cuda")
            position = (mask.cumsum(-1)-1).clamp_min(0)
            output = model(input_ids=padded, attention_mask=mask, position_ids=position,
                           use_cache=False, logits_to_keep=1).logits[:, -1]
            result[indexes] = output.float().log_softmax(-1).cpu().numpy()
    data = out / f"step_{step:03d}_logp.npy"
    tmp = data.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.save(f, result, allow_pickle=False)
    tmp.replace(data)
    atomic_json(out / f"step_{step:03d}_score.json", {
        "bank_sha256": sha256_file(bank), "checkpoint_manifest_sha256": sha256_file(path / "checkpoint_manifest.json"),
        "step": step, "row_count": len(rows), "vocabulary_size": result.shape[-1],
    })


def summarize(run, out):
    bank = out / "bank.json"
    rows = json.loads(bank.read_text())["rows"]
    steps = list(range(0, 101, 10))
    logp, validation = [], []
    for i in steps:
        tag = f"step_{i:03d}"
        identity = json.loads((out / f"{tag}_score.json").read_text())
        if identity["bank_sha256"] != sha256_file(bank) or identity["checkpoint_manifest_sha256"] != sha256_file(
            run / "exports" / tag / "checkpoint_manifest.json"):
            raise ValueError("Score provenance mismatch")
        logp.append(np.load(out / f"{tag}_logp.npy", mmap_mode="r", allow_pickle=False))
        validation.append(json.loads((run/"validation"/tag/"summary.json").read_text())["mean_reward"])
    extras = {field: np.asarray([r[key] for r in rows]) for field, key in
              (("observed_index", "observed_token"), ("trajectory_id", "trajectory_id"), ("trajectory_reward", "reward"))}
    intervals = {}
    for i, j in [(i, i+1) for i in range(10)] + [(0,4), (0,10), (5,10)]:
        intervals[f"{steps[i]}:{steps[j]}"] = teacher(logp, i, j, True, rewards=validation, **extras)
    atomic_json(out/"teacher_only.json", {"prefixes": len(rows), "trajectories": len(rows)//2,
                                        "bank_sha256": sha256_file(bank), "intervals": intervals})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze","score","summarize"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == "freeze":
        freeze(args.run, args.out)
    elif args.action == "score":
        if args.step is None:
            parser.error("score requires --step")
        score(args.run / "exports" / f"step_{args.step:03d}", args.out, args.step)
    else:
        summarize(args.run, args.out)


if __name__ == "__main__":
    main()
