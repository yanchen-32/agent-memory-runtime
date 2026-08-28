from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate used when no model tokenizer is configured."""
    return len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", text))


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[，。！？,.!?;；：:、（）()\[\]「」『』\"“”‘’]", "", text)


def answer_metrics(prediction: str, expected_answer: str) -> dict[str, object]:
    normalized_prediction = normalize_answer(prediction)
    normalized_expected = normalize_answer(expected_answer)
    return {
        "exact_match": normalized_prediction == normalized_expected,
        "normalized_prediction": normalized_prediction,
        "normalized_expected": normalized_expected,
    }


def retrieval_metrics(
    ranked_ids: list[str],
    expected_ids: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | None]:
    """Return retrieval metrics; None means retrieval is not applicable."""
    if not expected_ids:
        result: dict[str, float | None] = {}
        for k in ks:
            result[f"recall@{k}"] = None
            result[f"precision@{k}"] = None
        result["mrr"] = None
        return result

    expected = set(expected_ids)
    result = {}
    for k in ks:
        top = ranked_ids[:k]
        result[f"recall@{k}"] = (
            1.0 if any(x in expected for x in top) else 0.0
        )
        result[f"precision@{k}"] = sum(1 for x in top if x in expected) / k
    rr = 0.0
    for rank, memory_id in enumerate(ranked_ids, start=1):
        if memory_id in expected:
            rr = 1.0 / rank
            break
    result["mrr"] = rr
    return result
