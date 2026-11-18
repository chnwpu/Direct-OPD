"""Compute full-vocabulary model entropy over generated AIME response tokens."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    totals = defaultdict(lambda: {"tokens": 0, "entropy_sum": 0.0, "trajectories": 0})
    trajectories = torch.load(args.tokens, weights_only=True, map_location="cpu")
    with torch.inference_mode():
        for row in trajectories:
            prompt, response = row["prompt_ids"], row["response_ids"]
            if not response:
                continue
            ids = torch.tensor([prompt + response[:-1]], device="cuda")
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, len(prompt)-1:]
            total = totals[row["data_source"]]
            for block in hidden.split(64):
                logp = model.lm_head(block).float().log_softmax(-1)
                total["entropy_sum"] += float(-(logp.exp() * logp).sum())
            total["tokens"] += len(response)
            total["trajectories"] += 1
    for total in totals.values():
        total["mean_entropy_nats"] = total["entropy_sum"] / total["tokens"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(totals, indent=2) + "\n")


if __name__ == "__main__":
    main()
