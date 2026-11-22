"""Offline metrics on fixed-support, pre-scored Teacher data and optional Student probes.

NPZ inputs must carry prefix_ids, candidate_ids and teacher_logp arrays. Each
checkpoint is scored on identical prefixes and token IDs. Full-vocabulary KL
is only reported for explicitly complete distributions.
"""

import argparse
from pathlib import Path

import numpy as np

from grpo_dynamics.record import atomic_json


def probability(logp, full_vocab):
    if not np.isfinite(logp).all():
        raise ValueError("All log probabilities must be finite")
    p = np.exp(logp)
    mass = p.sum(axis=-1, keepdims=True)
    if (mass <= 0).any() or (mass > 1.001).any():
        raise ValueError("Invalid probability mass")
    if full_vocab and not np.allclose(mass, 1, atol=1e-5):
        raise ValueError("Full-vocabulary probabilities do not sum to 1")
    return p / mass


def shift(logp, i, j):
    if not 0 <= i < j < len(logp):
        raise ValueError("Expected ordered checkpoint indices i < j")
    return logp[j] - logp[i]


def average_by_trajectory(array, observed_index, trajectory_id):
    observed = np.take_along_axis(array, observed_index[:, None], axis=1).ravel()
    ids, inv = np.unique(trajectory_id, return_inverse=True)
    means = np.bincount(inv, weights=observed) / np.bincount(inv)
    return ids, means


def teacher(logp, i, j, full_vocab=False, rewards=None, observed_index=None,
            trajectory_id=None, trajectory_reward=None):
    d = shift(logp, i, j)
    old = probability(logp[i], full_vocab)
    new = probability(logp[j], full_vocab)
    initial = probability(logp[0], full_vocab)
    result = {
        "shift_rms": float(np.sqrt(np.mean(d*d))),
        "shift_l2_norm": float(np.linalg.norm(d)),
        "support": "full_vocabulary" if full_vocab else "renormalized_candidate_support",
        "teacher_kl_new_to_old": float(np.mean(np.sum(new * (np.log(new)-np.log(old)), axis=-1))),
        "reference_kl_new_to_initial": float(np.mean(np.sum(new * (np.log(new)-np.log(initial)), axis=-1))),
    }
    if rewards is not None:
        result["checkpoint_reward_gain"] = float(rewards[j] - rewards[i])
    if observed_index is not None:
        ids, mean = average_by_trajectory(d, observed_index, trajectory_id)
        labels = np.array([trajectory_reward[np.flatnonzero(trajectory_id == x)[0]] for x in ids])
        pos, neg = labels == 1, labels == 0
        result["reward_shift_alignment"] = float(mean[pos].mean()-mean[neg].mean()) if pos.any() and neg.any() else None
    return result


def student_probe(logp, i, j, student_logp, comparison=None, observed_index=None,
                  trajectory_id=None, trajectory_reward=None):
    d = shift(logp, i, j)
    w = probability(student_logp, False)
    result = {}
    if comparison is not None:
        k, l = comparison
        next_shift = shift(logp, k, l)
        centered = d - np.sum(w*d, axis=-1, keepdims=True)
        other = next_shift - np.sum(w*next_shift, axis=-1, keepdims=True)
        norm = np.sqrt(np.sum(w*centered**2) * np.sum(w*other**2))
        result["student_weighted_centered_cosine"] = float(np.sum(w*centered*other)/norm) if norm > 0 else None
    if observed_index is not None:
        ids, mean = average_by_trajectory(d, observed_index, trajectory_id)
        labels = np.array([trajectory_reward[np.flatnonzero(trajectory_id == x)[0]] for x in ids])
        result["reward_shift_covariance"] = float(np.mean((mean-mean.mean())*(labels-labels.mean())))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--student-probe", type=Path)
    p.add_argument("--interval", nargs=2, type=int, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = np.load(args.input, allow_pickle=False)
    probes = np.load(args.student_probe, allow_pickle=False) if args.student_probe else None
    steps = data["steps"].tolist()
    if len(set(steps)) != len(steps) or steps != sorted(steps):
        raise ValueError("Teacher steps must be unique and ordered")
    pairs = [(steps.index(i), steps.index(j)) for i, j in args.interval]
    def arrays(source):
        return {key: source[key] for key in ("observed_index","trajectory_id","trajectory_reward") if key in source}
    result = {}
    student_result = {}
    for m, (i, j) in enumerate(pairs):
        tag = f"{steps[i]}:{steps[j]}"
        result[tag] = teacher(data["teacher_logp"], i, j, bool(data.get("full_vocab", False)),
                              data.get("checkpoint_reward"), **arrays(data))
        if probes is not None:
            if probes["steps"].tolist() != steps:
                raise ValueError("Teacher and probe checkpoint labels differ")
            student_result[tag] = student_probe(probes["teacher_logp"], i, j, probes["student_logp"],
                          pairs[m+1] if m+1 < len(pairs) else None, **arrays(probes))
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out/"teacher_only.json", result)
    if probes is not None:
        atomic_json(args.out/"student_probe.json", student_result)


if __name__ == "__main__":
    main()
