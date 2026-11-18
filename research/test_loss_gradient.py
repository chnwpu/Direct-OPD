"""Check the official reward implementation and its on-policy logit gradient."""

import ast
import tempfile
from pathlib import Path

import torch

from research.gradient_capture import candidate_logit_gradient, capture_micro_batch


def official_reward_function():
    source = Path(__file__).parents[1] / "verl/verl/workers/actor/dp_actor.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_delta_opd_rm_scores")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_compute_delta_opd_rm_scores"]


def test_direct_opd_topk_gradient() -> None:
    logits = torch.tensor([0.4, -0.2, -1.0, -2.0], dtype=torch.float64, requires_grad=True)
    full_logp = logits.log_softmax(dim=-1)
    ids = torch.tensor([0, 1])
    t0 = torch.tensor([0.001, 0.15], dtype=torch.float64)
    t1 = torch.tensor([0.1, 0.89], dtype=torch.float64)
    official = official_reward_function()
    advantage_3d, reward_3d = official(
        full_logp[ids].view(1, 1, 2), t1.log().view(1, 1, 2), t0.log().view(1, 1, 2)
    )
    advantage = advantage_3d.flatten()
    reward = reward_3d.flatten()
    weight = full_logp[ids].detach().softmax(dim=-1)
    torch.testing.assert_close(advantage, weight * reward)
    assert not advantage.requires_grad
    loss = -(advantage * full_logp[ids]).sum()
    loss.backward()

    p = full_logp.detach().exp()
    expected = p * advantage.sum()
    expected[ids] -= advantage
    torch.testing.assert_close(logits.grad, expected)
    torch.testing.assert_close(p.sum(), torch.tensor(1.0, dtype=p.dtype))
    assert abs(reward[0].item() - torch.tensor(100.0).log().item()) < 1e-6
    assert expected[0].abs().item() != (p[0] * reward[0]).item()


def test_official_ppo_and_adaptive_kl_gradient() -> None:
    """Check clipped top-k PPO plus the sampled low-var KL in the actor loss."""
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, kl_penalty

    ids = torch.tensor([0, 1])
    sampled_id = 1
    advantages = torch.tensor([0.3, -0.2], dtype=torch.float64).view(1, 1, 2)
    config = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": None, "clip_ratio_high": None, "clip_ratio_c": 3.0})
    coefficient = 1.7
    for ratio_values, clipped in [([2.7, 0.37], True), ([1.1, 0.9], False)]:
        logits = torch.tensor([0.4, -0.2, -1.0, -2.0], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(dim=-1)
        ratios = torch.tensor(ratio_values, dtype=torch.float64)
        old_logp = (logp[ids].detach() - ratios.log()).view(1, 1, 2)
        topk_logp = logp[ids].view(1, 1, 2)
        sampled_logp = logp[sampled_id].view(1, 1)
        pg_loss, _ = compute_policy_loss_vanilla(
            old_log_prob=old_logp,
            log_prob=topk_logp,
            advantages=advantages,
            response_mask=torch.ones((1, 1), dtype=torch.float64),
            loss_agg_mode="token-mean",
            config=config,
        )
        ref_logp = logp[sampled_id].detach() - 0.2
        loss = pg_loss + coefficient * kl_penalty(sampled_logp, ref_logp, "low_var_kl").mean()
        with tempfile.TemporaryDirectory() as temp_dir:
            capture_micro_batch(
                loss=loss,
                topk_logp=topk_logp,
                sampled_logp=sampled_logp,
                model_inputs={
                    "diagnostic_position_mask": torch.ones((1, 1), dtype=torch.bool),
                    "diagnostic_trace_id": torch.tensor([0]),
                    "student_top_k_ids": ids.view(1, 1, 2),
                    "responses": torch.tensor([[sampled_id]]),
                    "student_top_k_log_probs": old_logp,
                    "advantages": advantages,
                    "ref_log_prob": ref_logp.view(1, 1),
                },
                output_dir=temp_dir,
                step=10,
                mini_batch_index=0,
                micro_batch_index=0,
            )
            captured = torch.load(next(Path(temp_dir).glob("gradient_*.pt")), weights_only=True)
        loss.backward()

        # For this sign pattern, both candidates are clipped in the first case.
        pg_derivatives = torch.zeros_like(ratios) if clipped else -advantages.flatten() * ratios
        sampled_derivative = coefficient * (1 - torch.exp(ref_logp - logp[sampled_id].detach()))
        total_derivative = pg_derivatives.sum() + sampled_derivative
        expected = -logp.detach().exp() * total_derivative
        expected[ids] += pg_derivatives
        expected[sampled_id] += sampled_derivative
        torch.testing.assert_close(logits.grad, expected)
        torch.testing.assert_close(captured["d_actual_scaled_loss_d_candidate_logit"], expected[ids].view(1, 2))
        torch.testing.assert_close(
            candidate_logit_gradient(
                logp[ids].detach().unsqueeze(0), pg_derivatives.unsqueeze(0), sampled_derivative.view(1),
                ids.unsqueeze(0), torch.tensor([sampled_id]),
            ),
            expected[ids].unsqueeze(0),
        )


if __name__ == "__main__":
    test_direct_opd_topk_gradient()
    test_official_ppo_and_adaptive_kl_gradient()
    print("Direct-OPD reward and official PPO+KL gradient tests passed")
