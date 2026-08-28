from __future__ import annotations


def retrieval_metrics(ranked_ids: list[str], expected_ids: list[str], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    expected = set(expected_ids)
    result: dict[str, float] = {}
    for k in ks:
        top = ranked_ids[:k]
        result[f"recall@{k}"] = 1.0 if expected and any(x in expected for x in top) else 0.0
        if k > 0:
            result[f"precision@{k}"] = sum(1 for x in top if x in expected) / k
    rr = 0.0
    for rank, memory_id in enumerate(ranked_ids, start=1):
        if memory_id in expected:
            rr = 1.0 / rank
            break
    result["mrr"] = rr
    return result
