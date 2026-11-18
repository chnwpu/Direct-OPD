"""Additional read-only metrics for training rollouts."""


def repetition_4gram_rate(responses, response_mask) -> float:
    """Mean fraction of response 4-grams that repeat an earlier 4-gram."""
    values = []
    for ids, mask in zip(responses.cpu().tolist(), response_mask.cpu().tolist()):
        valid = [token for token, keep in zip(ids, mask) if keep]
        if len(valid) < 4:
            continue
        seen = set()
        repeated = 0
        for offset in range(len(valid) - 3):
            gram = tuple(valid[offset : offset + 4])
            repeated += gram in seen
            seen.add(gram)
        values.append(repeated / (len(valid) - 3))
    return sum(values) / len(values) if values else 0.0


def validation_metrics(sources, uids, scores, lengths, repetitions, max_length):
    from collections import defaultdict
    from math import comb

    groups = defaultdict(lambda: defaultdict(list))
    for source, uid, score, length, repeat in zip(sources, uids, scores, lengths, repetitions):
        groups[source][uid].append((score > 0, length, repeat))

    metrics = {}
    for source, prompts in groups.items():
        prefix = f"val-diagnostics/{source}"
        rows = [row for samples in prompts.values() for row in samples]
        metrics[f"{prefix}/avg_accuracy"] = sum(row[0] for row in rows) / len(rows)
        metrics[f"{prefix}/mean_length"] = sum(row[1] for row in rows) / len(rows)
        metrics[f"{prefix}/truncation_rate"] = sum(row[1] >= max_length for row in rows) / len(rows)
        metrics[f"{prefix}/repetition_4gram_rate"] = sum(row[2] for row in rows) / len(rows)
        for k in (1, 2, 4, 8, 16, 32):
            if k <= min(map(len, prompts.values())):
                metrics[f"{prefix}/pass@{k}"] = sum(
                    1 - comb(len(samples) - sum(row[0] for row in samples), k) / comb(len(samples), k)
                    for samples in prompts.values()
                ) / len(prompts)
    return metrics
