"""Freeze a balanced prefix bank and score the same candidates for every model."""

import argparse
from itertools import groupby
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def score(model_path, rows, batch_size, fixed_candidates):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    outputs = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prefix_ids"]))
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            prefixes = [rows[i]["prefix_ids"] for i in indices]
            width = len(prefixes[-1])
            ids = torch.tensor([[0] * (width - len(p)) + p for p in prefixes], device="cuda")
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prefixes], device="cuda")
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           use_cache=False, logits_to_keep=1).logits[:, -1].float()
            logp = logits.log_softmax(-1)
            top = logits.topk(16).indices
            for j, i in enumerate(indices):
                candidates = torch.as_tensor(rows[i]["candidate_ids"], device="cuda") if fixed_candidates else top[j]
                values = logp[j, candidates]
                rank = (logits[j, :, None] > logits[j, candidates]).sum(0) + 1
                outputs[i] = {
                    "candidate_ids": candidates.tolist(),
                    "logp": values.tolist(),
                    "rank": rank.tolist(),
                    "top1": int(top[j, 0]),
                    "entropy": float(-(logp[j].exp() * logp[j]).sum()),
                }
    return outputs


def select(rows, scores):
    def stratum(index):
        row = rows[index]
        return row["data_source"], row["correct"], row["relative_position"]

    selected = []
    for key, group in groupby(sorted(range(len(rows)), key=stratum), key=stratum):
        ordered = sorted(group, key=lambda i: (scores[i]["entropy"], i))
        if len(ordered) != 128:
            raise ValueError(f"Expected 128 candidates per benchmark/accuracy/position stratum: {key}")
        for entropy_quartile in range(4):
            start = entropy_quartile * 32
            for i in ordered[start:start + 32:2]:
                selected.append({**rows[i], "candidate_ids": scores[i]["candidate_ids"],
                                 "entropy_quartile": entropy_quartile, "student0": scores[i]})
    if len(selected) != 1024:
        raise ValueError(f"Expected 1024 bank entries, got {len(selected)}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "score"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    data = torch.load(args.input, map_location="cpu", weights_only=True)
    freeze = args.mode == "freeze"
    rows = data["candidates" if freeze else "prefixes"]
    results = score(args.model, rows, args.batch_size, fixed_candidates=not freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"prefixes": select(rows, results)} if freeze else {"scores": results, "model": str(args.model)}
    temp = args.output.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(args.output)


if __name__ == "__main__":
    main()
