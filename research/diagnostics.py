"""Read-only, deterministic snapshots of real on-policy Direct-OPD batches."""

from pathlib import Path

import numpy as np
import torch


def dump_topk_batch(batch, step: int, output_dir: str, trajectories: int = 32, positions: int = 128) -> Path:
    """Persist all data needed to rescore selected prefixes without changing training."""
    tensors = batch.batch
    required = (
        "input_ids", "attention_mask", "responses", "response_mask",
        "student_top_k_ids", "student_top_k_log_probs",
        "teacher_on_student_log_probs", "teacher_ref_on_student_log_probs",
    )
    missing = [name for name in required if name not in tensors]
    if missing:
        raise KeyError(f"diagnostic batch is missing {missing}")

    n = tensors["responses"].shape[0]
    response_len = tensors["responses"].shape[1]
    prompt_len = tensors["input_ids"].shape[1] - response_len
    selected = np.linspace(0, n - 1, min(trajectories, n), dtype=np.int64)
    records = []
    for batch_index in selected:
        i = int(batch_index)
        mask = tensors["response_mask"][i].bool().cpu()
        valid = torch.nonzero(mask, as_tuple=False).flatten()
        if len(valid) == 0:
            continue
        pick = torch.linspace(0, len(valid) - 1, min(positions, len(valid))).long()
        token_positions = valid[pick]
        student_logp = tensors["student_top_k_log_probs"][i, token_positions].detach().cpu().float()
        t1_logp = tensors["teacher_on_student_log_probs"][i, token_positions].detach().cpu().float()
        t0_logp = tensors["teacher_ref_on_student_log_probs"][i, token_positions].detach().cpu().float()
        shift = t1_logp - t0_logp
        weighted_shift = student_logp.softmax(dim=-1) * shift
        candidate_ids = tensors["student_top_k_ids"][i, token_positions].detach().cpu()
        sampled_ids = tensors["responses"][i, token_positions].detach().cpu()
        records.append({
            "batch_index": i,
            "input_ids": tensors["input_ids"][i].detach().cpu(),
            "attention_mask": tensors["attention_mask"][i].detach().cpu(),
            "response_mask": mask,
            "response_ids": tensors["responses"][i].detach().cpu(),
            "token_positions": token_positions,
            "candidate_ids": candidate_ids,
            "student_logp_full_vocab": student_logp,
            "t1_logp_full_vocab": t1_logp,
            "t0_logp_full_vocab": t0_logp,
            "policy_shift": shift,
            "weighted_policy_shift": weighted_shift,
            "sampled_token_flag": candidate_ids.eq(sampled_ids[:, None]),
            "student_reference_sampled_logp": (
                tensors["ref_log_prob"][i, token_positions].detach().cpu().float()
                if "ref_log_prob" in tensors else None
            ),
            "prompt_length": prompt_len,
        })

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"on_policy_step_{step:03d}.pt"
    torch.save({"step": step, "batch_size": n, "records": records}, path)
    return path
