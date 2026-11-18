"""Resolve all T1 ranks on the saved on-policy trajectories."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    with torch.inference_mode():
        for path in sorted(args.diagnostic_dir.glob("on_policy_step_*.pt")):
            payload = torch.load(path, weights_only=True, map_location="cpu")
            ranks = []
            for row in payload["records"]:
                prompt_end = row["prompt_length"]
                prompt = row["input_ids"][:prompt_end][row["attention_mask"][:prompt_end].bool()]
                response = row["response_ids"][:int(row["token_positions"][-1])]
                input_ids = torch.cat((prompt, response)).unsqueeze(0).cuda()
                positions = (len(prompt) + row["token_positions"] - 1).cuda()
                logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=positions).logits[0].float()
                candidates = row["candidate_ids"].cuda()
                candidate_logits = logits.gather(-1, candidates)
                rank = torch.stack([
                    (logits[:, start:start+4096, None] > candidate_logits[:, None, :]).sum(1)
                    for start in range(0, logits.shape[-1], 4096)
                ]).sum(0) + 1
                ranks.append({
                    "batch_index": row["batch_index"],
                    "token_positions": row["token_positions"],
                    "candidate_ids": row["candidate_ids"],
                    "teacher_t1_rank": rank.cpu(),
                })
            destination = path.with_name(path.name.replace("on_policy_", "teacher_ranks_"))
            tmp = destination.with_suffix(".tmp")
            torch.save({"step": payload["step"], "records": ranks}, tmp)
            tmp.replace(destination)


if __name__ == "__main__":
    main()
