from __future__ import annotations

from collections import Counter
import re
import unicodedata


_ANSWER_PREFIX_RE = re.compile(
    r"^(?:答案|回答|结果|结论)(?:应当|应该|为|是|：|:)*\s*",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", flags=re.IGNORECASE)


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate used when no model tokenizer is configured."""
    return len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", text))


def normalize_answer(text: str) -> str:
    """Apply the protocol's frozen, deterministic answer normalization."""
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = _ANSWER_PREFIX_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    return "".join(character for character in text if not _is_punctuation(character))


def _is_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith("P")


def answer_tokens(text: str) -> list[str]:
    """Tokenize English/digits by word and Chinese deterministically by character."""
    normalized = unicodedata.normalize("NFKC", str(text)).strip().lower()
    normalized = _ANSWER_PREFIX_RE.sub("", normalized)
    return _TOKEN_RE.findall(normalized)


def _answer_candidate(text: str) -> str:
    """Extract only a frozen, explicitly wrapped first answer clause."""
    normalized = unicodedata.normalize("NFKC", str(text)).strip().lower()
    prefix = _ANSWER_PREFIX_RE.match(normalized)
    if prefix is None:
        return normalize_answer(normalized)
    candidate = normalized[prefix.end():]
    candidate = re.split(
        r"(?:[。！？\n;；]|[,，]?(?:因为|原因是|理由是|because|since))",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return normalize_answer(candidate)


def answer_metrics(
    prediction: str,
    expected_answer: str,
    answer_aliases: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Return raw/normalized matches and token-overlap Answer F1.

    Aliases are benchmark-owned ground truth. The best matching frozen target
    is used for normalized match and F1; predictions never create new aliases.
    """
    normalized_prediction = normalize_answer(prediction)
    targets = [str(expected_answer), *(answer_aliases or [])]
    normalized_targets = [normalize_answer(target) for target in targets]

    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_target = targets[0]
    prediction_tokens = answer_tokens(prediction)
    for target in targets:
        expected_tokens = answer_tokens(target)
        if not prediction_tokens and not expected_tokens:
            precision = recall = f1 = 1.0
        elif not prediction_tokens or not expected_tokens:
            precision = recall = f1 = 0.0
        else:
            overlap = sum((Counter(prediction_tokens) & Counter(expected_tokens)).values())
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(expected_tokens)
            f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
        if (f1, recall, precision) > (best_f1, best_recall, best_precision):
            best_precision, best_recall, best_f1 = precision, recall, f1
            best_target = target

    normalized_match = normalized_prediction in normalized_targets
    answer_candidate = _answer_candidate(prediction)
    answer_match = answer_candidate in normalized_targets
    return {
        "exact_match": str(prediction) == str(expected_answer),
        "normalized_match": normalized_match,
        "answer_match": answer_match,
        "answer_accuracy": int(answer_match),
        "answer_precision": best_precision,
        "answer_recall": best_recall,
        "answer_f1": best_f1,
        "normalized_prediction": normalized_prediction,
        "answer_candidate": answer_candidate,
        "normalized_expected": normalize_answer(best_target),
        "matched_target": best_target,
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
        result[f"recall@{k}"] = len(set(top) & expected) / len(expected)
        result[f"precision@{k}"] = sum(1 for x in top if x in expected) / k
    rr = 0.0
    for rank, memory_id in enumerate(ranked_ids, start=1):
        if memory_id in expected:
            rr = 1.0 / rank
            break
    result["mrr"] = rr
    return result
