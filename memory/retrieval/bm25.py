from __future__ import annotations

import math
import re
from collections import Counter

from memory.schema import SearchResult
from memory.storage import MemoryStore


def tokenize(text: str) -> list[str]:
    ascii_tokens = re.findall(r"[A-Za-z0-9_.+-]+", text.lower())
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    zh_bigrams = ["".join(chars[i:i+2]) for i in range(max(0, len(chars) - 1))]
    return ascii_tokens + chars + zh_bigrams


class BM25Retriever:
    def __init__(self, store: MemoryStore, k1: float = 1.5, b: float = 0.75):
        self.store = store
        self.k1 = k1
        self.b = b

    def search(self, query: str, top_k: int = 5, user_id: str | None = None) -> list[SearchResult]:
        records = self.store.list_active(user_id=user_id)
        if top_k <= 0 or not records:
            return []
        docs = [tokenize(r.content) for r in records]
        query_terms = tokenize(query)
        avgdl = sum(len(d) for d in docs) / max(1, len(docs))
        df = Counter(term for term in set(query_terms) for doc in docs if term in doc)
        scores: list[float] = []
        n = len(docs)
        for doc in docs:
            tf = Counter(doc)
            score = 0.0
            for term in query_terms:
                if tf[term] == 0:
                    continue
                idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
                denom = tf[term] + self.k1 * (1 - self.b + self.b * len(doc) / max(avgdl, 1e-9))
                score += idf * tf[term] * (self.k1 + 1) / denom
            scores.append(score)
        order = sorted(range(n), key=lambda i: (-scores[i], records[i].memory_id))[: min(top_k, n)]
        return [
            SearchResult(
                memory_id=records[i].memory_id,
                content=records[i].content,
                score=float(scores[i]),
                metadata={"retriever": "bm25-v1", "memory_type": records[i].memory_type.value},
            )
            for i in order
        ]
