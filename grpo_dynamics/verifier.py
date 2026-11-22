"""Binary math verifier shared by training, preflight and validation."""

import re
from functools import lru_cache

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

EXTRACT = (LatexExtractionConfig(), ExprExtractionConfig())


def last_box(text):
    # Walk balanced braces so fractions and nested expressions remain intact.
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        depth = 1
        for pos in range(match.end(), len(text)):
            if pos and text[pos-1] == "\\":
                continue
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[match.end():pos]
    return None


@lru_cache(maxsize=8192)
def parsed_gold(text):
    stripped = text.strip()
    if not stripped:
        return []
    return parse("$" + (last_box(stripped) or stripped.strip("$")) + "$",
                 extraction_config=EXTRACT, fallback_mode="no_fallback",
                 extraction_mode="first_match", parsing_timeout=4)


def score_answer(response, ground_truth):
    text = str(response).strip()
    boxed = last_box(text)
    if boxed is not None:
        candidate = boxed.strip()
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidate = re.sub(r"^(?:final answer|answer)\s*:\s*", "", lines[-1], flags=re.I) if lines else ""
    if not candidate:
        return {"score": 0.0, "acc": False, "extracted_answer": ""}
    gold = parsed_gold(str(ground_truth))
    prediction = parse("$" + candidate.strip("$") + "$",
                       extraction_config=EXTRACT, fallback_mode="no_fallback",
                       extraction_mode="first_match", parsing_timeout=4)
    result = bool(gold and prediction and verify(gold, prediction, timeout_seconds=4))
    return {"score": float(result), "acc": result, "extracted_answer": candidate}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return score_answer(solution_str, ground_truth)
