"""Capture the actual actor-loss derivative on selected response positions."""

from pathlib import Path

import torch


def candidate_logit_gradient(
    topk_logp: torch.Tensor,
    topk_logp_derivative: torch.Tensor,
    sampled_logp_derivative: torch.Tensor,
    candidate_ids: torch.Tensor,
    sampled_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply the full-vocabulary log-softmax Jacobian to the official loss derivative."""
    total = topk_logp_derivative.sum(dim=-1, keepdim=True) + sampled_logp_derivative.unsqueeze(-1)
    return (
        topk_logp_derivative
        + candidate_ids.eq(sampled_ids.unsqueeze(-1)) * sampled_logp_derivative.unsqueeze(-1)
        - topk_logp.exp() * total
    )


def capture_micro_batch(
    *,
    loss: torch.Tensor,
    topk_logp: torch.Tensor,
    sampled_logp: torch.Tensor,
    model_inputs: dict,
    output_dir: str,
    step: int,
    mini_batch_index: int,
    micro_batch_index: int,
) -> None:
    """Record d(actual scaled PPO+KL loss)/d logit without a second model backward."""
    mask = model_inputs["diagnostic_position_mask"].bool()
    rows, positions = torch.nonzero(mask, as_tuple=True)
    if rows.numel() == 0:
        return
    topk_derivative, sampled_derivative = torch.autograd.grad(
        loss, (topk_logp, sampled_logp), retain_graph=True, allow_unused=True
    )
    if sampled_derivative is None:
        sampled_derivative = torch.zeros_like(sampled_logp)
    ids = model_inputs["student_top_k_ids"][rows, positions]
    sampled_ids = model_inputs["responses"][rows, positions]
    candidate_gradient = candidate_logit_gradient(
        topk_logp[rows, positions],
        topk_derivative[rows, positions],
        sampled_derivative[rows, positions],
        ids,
        sampled_ids,
    )
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (
        f"gradient_step_{step:03d}_rank_{rank:02d}_"
        f"mini_{mini_batch_index:02d}_micro_{micro_batch_index:03d}.pt"
    )
    torch.save(
        {
            "step": step,
            "rank": rank,
            "trajectory_index": model_inputs["diagnostic_trace_id"][rows].detach().cpu(),
            "token_position": positions.detach().cpu(),
            "candidate_ids": ids.detach().cpu(),
            "sampled_ids": sampled_ids.detach().cpu(),
            "current_student_logp_full_vocab": topk_logp[rows, positions].detach().cpu(),
            "current_sampled_logp": sampled_logp[rows, positions].detach().cpu(),
            "old_student_logp_full_vocab": model_inputs["student_top_k_log_probs"][rows, positions].detach().cpu(),
            "advantages": model_inputs["advantages"][rows, positions].detach().cpu(),
            "sampled_ref_logp": (
                model_inputs["ref_log_prob"][rows, positions].detach().cpu()
                if "ref_log_prob" in model_inputs else None
            ),
            "d_actual_scaled_loss_d_candidate_logit": candidate_gradient.detach().cpu(),
            "d_actual_scaled_loss_d_candidate_logp": topk_derivative[rows, positions].detach().cpu(),
            "d_actual_scaled_loss_d_sampled_logp": sampled_derivative[rows, positions].detach().cpu(),
        },
        path,
    )
