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
