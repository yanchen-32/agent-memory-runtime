from __future__ import annotations

from dataclasses import dataclass, field
import re

from memory.schema import SearchResult
from memory.storage import MemoryStore


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", text))


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(slots=True)
class BudgetSelection:
    selected: list[SearchResult]
    selected_indices: list[int]
    tokens_before: int
    tokens_after: int
    budget: int | None
    score_details: dict[int, dict[str, float]] = field(default_factory=dict)

    @property
    def savings_tokens(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


class ContextBudgetManager:
    """Greedy budget-aware selection with interpretable feature scores.

    The budget applies to the complete prompt represented by prefix, selected
    context lines and suffix. The manager first ranks candidates by relevance,
    importance, diversity and token efficiency, then keeps only candidates that
    fit the budget.
    """

    def __init__(
        self,
        store: MemoryStore,
        relevance_weight: float = 0.40,
        importance_weight: float = 0.20,
        diversity_weight: float = 0.15,
        redundancy_weight: float = 0.15,
        token_cost_weight: float = 0.10,
    ):
        self.store = store
        self.weights = {
            "relevance": relevance_weight,
            "importance": importance_weight,
            "diversity": diversity_weight,
            "redundancy": redundancy_weight,
            "token_efficiency": token_cost_weight,
        }

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if high - low < 1e-12:
            return [1.0] * len(values)
        return [(value - low) / (high - low) for value in values]

    def select(
        self,
        query: str,
        candidates: list[SearchResult],
        context_lines: list[str],
        token_budget: int | None,
        prefix: str = "",
        suffix: str = "",
    ) -> BudgetSelection:
        if len(candidates) != len(context_lines):
            raise ValueError("candidates and context_lines must have equal length")

        all_prompt = prefix + "\n".join(context_lines) + suffix
        tokens_before = estimate_tokens(all_prompt)
        if not candidates or token_budget is None:
            return BudgetSelection(
                selected=list(candidates),
                selected_indices=list(range(len(candidates))),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                budget=token_budget,
            )
        if token_budget <= 0:
            return BudgetSelection([], [], tokens_before, estimate_tokens(prefix + suffix), token_budget)

        relevance = self._normalize([candidate.score for candidate in candidates])
        query_tokens = _token_set(query)
        line_tokens = [_token_set(line) for line in context_lines]
        costs = [max(1, estimate_tokens(line)) for line in context_lines]
        max_cost = max(costs)
        importance: list[float] = []
        for candidate in candidates:
            record = self.store.get(candidate.memory_id)
            importance.append(record.importance if record is not None else 0.0)

        selected_indices: list[int] = []
        selected_tokens: list[str] = []
        details: dict[int, dict[str, float]] = {}
        remaining = set(range(len(candidates)))

        while remaining:
            scored: list[tuple[float, int]] = []
            for index in remaining:
                redundancy = max(
                    (_jaccard(line_tokens[index], line_tokens[j]) for j in selected_indices),
                    default=0.0,
                )
                diversity = 1.0 - redundancy
                query_overlap = (
                    len(line_tokens[index] & query_tokens) / max(1, len(query_tokens))
                )
                token_efficiency = 1.0 - (costs[index] / max_cost)
                base_score = (
                    self.weights["relevance"] * relevance[index]
                    + self.weights["importance"] * importance[index]
                    + self.weights["diversity"] * diversity
                    + self.weights["redundancy"] * (1.0 - redundancy)
                    + self.weights["token_efficiency"] * token_efficiency
                )
                utility = base_score / (costs[index] ** 0.25)
                details[index] = {
                    "relevance": relevance[index],
                    "importance": importance[index],
                    "query_overlap": query_overlap,
                    "diversity": diversity,
                    "redundancy": redundancy,
                    "token_efficiency": token_efficiency,
                    "utility": utility,
                }
                scored.append((utility, index))

            chosen: int | None = None
            for _, index in sorted(scored, key=lambda item: (-item[0], item[1])):
                candidate_lines = [context_lines[i] for i in selected_indices] + [context_lines[index]]
                prompt = prefix + "\n".join(candidate_lines) + suffix
                if estimate_tokens(prompt) <= token_budget:
                    chosen = index
                    break
            if chosen is None:
                break
            selected_indices.append(chosen)
            selected_tokens.append(context_lines[chosen])
            remaining.remove(chosen)

        selected_indices.sort()
        selected = [candidates[index] for index in selected_indices]
        selected_prompt = prefix + "\n".join(selected_tokens) + suffix
        return BudgetSelection(
            selected=selected,
            selected_indices=selected_indices,
            tokens_before=tokens_before,
            tokens_after=estimate_tokens(selected_prompt),
            budget=token_budget,
            score_details=details,
        )
