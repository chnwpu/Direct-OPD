"""Verify and record each inference export after the pinned verl merger runs."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_dynamics.prepare import sha256_file
from grpo_dynamics.record import atomic_json


def tensors(path):
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found at {path}")
    result = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as data:
            for key in data.keys():
                if key in result:
                    raise ValueError(f"Duplicate weight: {key}")
                result[key] = file
    return result


def changed_weights(initial, exported):
    before, after = tensors(initial), tensors(exported)
    if set(before) != set(after):
        raise ValueError("Original and exported models have different weight keys")
    changed = 0
    for key in sorted(before):
        with safe_open(before[key], framework="pt", device="cpu") as a, safe_open(after[key], framework="pt", device="cpu") as b:
            x, y = a.get_tensor(key), b.get_tensor(key)
            if x.shape != y.shape or y.dtype != torch.bfloat16:
                raise ValueError(f"Shape or exported BF16 mismatch: {key}")
            changed += not torch.equal(x.to(torch.bfloat16), y)
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--initial-model", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--compare-original", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(6)
    launch = json.loads((args.run / "launch.json").read_text())
    metrics = (json.loads((args.run / "steps" / f"step_{args.step:03}.json").read_text())
               if args.step else {"cumulative_generated_tokens": 0, "cumulative_prompt_count": 0})
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer.apply_chat_template([{"role":"user","content":"What is 2+3?"}],
        enable_thinking=False, add_generation_prompt=True, return_tensors="pt")
    decoded = tokenizer.decode(ids[0], skip_special_tokens=False)
    assert "<think>\n\n</think>" in decoded, decoded
    with torch.inference_mode():
        result = model.generate(ids, do_sample=False, max_new_tokens=1)
    assert result.shape[1] == ids.shape[1] + 1
    config = args.run / "logs/resolved_config.yaml"
    metadata = {
        "step": args.step, "base_revision": launch["model_revision"],
        "cumulative_generated_tokens": metrics["cumulative_generated_tokens"],
        "cumulative_prompt_count": metrics["cumulative_prompt_count"],
        "data_manifest_sha256": launch["data_manifest_sha256"], "config_sha256": sha256_file(config),
        "protocol_sha256": launch["protocol_sha256"], "inference_verified": True,
        "test_generated_token_id": int(result[0, -1]),
        "files_sha256": {p.name: sha256_file(p) for p in sorted(args.model.iterdir())
                         if p.is_file() and p.name != "checkpoint_manifest.json"},
    }
    if args.compare_original:
        metadata["changed_weight_tensor_count"] = changed_weights(args.initial_model, args.model)
    destination = args.run / "exports" / f"step_{args.step:03}"
    # For Step 0 the weights are a symlink inside the export directory.
    atomic_json(destination / "checkpoint_manifest.json", metadata)
    print(f"Verified checkpoint {args.step} at {destination}")


if __name__ == "__main__":
    main()
