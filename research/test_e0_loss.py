"""Check the exact Student-top-k reverse KL used by the E0 control."""

import ast
from pathlib import Path

import torch


def official_e0_loss():
    source = Path(__file__).parents[1] / "verl/verl/workers/actor/dp_actor.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_exact_topk_reverse_kl"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_exact_topk_reverse_kl"]


def test_exact_reverse_kl_gradient() -> None:
    logits = torch.tensor([0.4, -0.2, -1.0, -2.0], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 1])
    teacher_probabilities = torch.tensor([0.1, 0.89], dtype=torch.float64)
    student_logp_full = logits.log_softmax(dim=-1)[ids]
    loss = official_e0_loss()(student_logp_full, teacher_probabilities.log())
    loss.backward()

    p = student_logp_full.detach().softmax(dim=-1)
    q = teacher_probabilities / teacher_probabilities.sum()
    local_kl = (p * (p.log() - q.log())).sum()
    expected = torch.zeros_like(logits)
    expected[ids] = p * (p.log() - q.log() - local_kl)
    torch.testing.assert_close(logits.grad, expected)
    assert loss.item() >= 0
    assert logits.grad[2:].abs().max().item() < 1e-12


if __name__ == "__main__":
    test_exact_reverse_kl_gradient()
    print("Exact E0 top-k reverse KL gradient test passed")
