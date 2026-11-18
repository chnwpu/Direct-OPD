"""Read-only, deterministic snapshots of real on-policy Direct-OPD batches."""

from pathlib import Path

import numpy as np
import torch


def select_positions(batch, trajectories: int = 32, positions: int = 128):
    """Select the same trajectories and positions for snapshots and gradient capture."""
    response_mask = batch.batch["response_mask"].bool().cpu()
    n, response_len = response_mask.shape
    chosen = np.linspace(0, n - 1, min(trajectories, n), dtype=np.int64)
    selected_mask = torch.zeros((n, response_len), dtype=torch.bool)
    for i in chosen:
        valid = torch.nonzero(response_mask[int(i)], as_tuple=False).flatten()
        if len(valid):
            pick = torch.linspace(0, len(valid) - 1, min(positions, len(valid))).long()
            selected_mask[int(i), valid[pick]] = True
    return selected_mask, torch.arange(n, dtype=torch.long)


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
    selected_mask, _ = select_positions(batch, trajectories, positions)
    records = []
    for batch_index in selected:
        i = int(batch_index)
        mask = tensors["response_mask"][i].bool().cpu()
        valid = torch.nonzero(mask, as_tuple=False).flatten()
        if len(valid) == 0:
            continue
        token_positions = torch.nonzero(selected_mask[i], as_tuple=False).flatten()
        student_logp = tensors["student_top_k_log_probs"][i, token_positions].detach().cpu().float()
        t1_logp = tensors["teacher_on_student_log_probs"][i, token_positions].detach().cpu().float()
        t0_logp = tensors["teacher_ref_on_student_log_probs"][i, token_positions].detach().cpu().float()
        shift = t1_logp - t0_logp
        weighted_shift = student_logp.softmax(dim=-1) * shift
        candidate_ids = tensors["student_top_k_ids"][i, token_positions].detach().cpu()
        sampled_ids = tensors["responses"][i, token_positions].detach().cpu()
        teacher_topk_ids = (
            tensors["teacher_top_k_ids"][i, token_positions].detach().cpu()
            if "teacher_top_k_ids" in tensors else None
        )
        teacher_rank_within_topk = None
        if teacher_topk_ids is not None:
            matches = candidate_ids.unsqueeze(-1).eq(teacher_topk_ids.unsqueeze(-2))
            teacher_rank_within_topk = torch.where(
                matches.any(dim=-1), matches.long().argmax(dim=-1) + 1, 0
            )
        records.append({
            "batch_index": i,
            "input_ids": tensors["input_ids"][i].detach().cpu(),
            "attention_mask": tensors["attention_mask"][i].detach().cpu(),
            "response_mask": mask,
            "response_ids": tensors["responses"][i].detach().cpu(),
            "token_positions": token_positions,
            "candidate_ids": candidate_ids,
            "student_logp_full_vocab": student_logp,
            "student_probability_full_vocab": student_logp.exp(),
            "t1_logp_full_vocab": t1_logp,
            "t1_probability_full_vocab": t1_logp.exp(),
            "t0_logp_full_vocab": t0_logp,
            "t0_probability_full_vocab": t0_logp.exp(),
            "policy_shift": shift,
            "weighted_policy_shift": weighted_shift,
            "weighted_policy_shift_topk_normalized": weighted_shift,
            "sampled_token_flag": candidate_ids.eq(sampled_ids[:, None]),
            "teacher_t1_topk_ids": teacher_topk_ids,
            "teacher_t1_rank_within_topk_or_zero": teacher_rank_within_topk,
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
