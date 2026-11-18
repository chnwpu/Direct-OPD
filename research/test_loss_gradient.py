"""Check the official reward implementation and its on-policy logit gradient."""

import ast
from pathlib import Path

import torch


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


if __name__ == "__main__":
    test_direct_opd_topk_gradient()
    print("Direct-OPD top-k gradient test passed")
