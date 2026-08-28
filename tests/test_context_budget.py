from memory import (
    ContextBudgetManager,
    HashEmbeddingModel,
    InMemoryMemoryStore,
    MemoryRecord,
    SearchResult,
)


def test_context_budget_respects_total_prompt_budget():
    store = InMemoryMemoryStore()
    store.add(MemoryRecord(memory_id="a", content="项目数据库使用 openGauss。", importance=0.9))
    store.add(MemoryRecord(memory_id="b", content="项目部署平台采用鲲鹏 ARM64。", importance=0.8))
    store.add(MemoryRecord(memory_id="c", content="今天下午天气很好。", importance=0.1))

    candidates = [
        SearchResult("a", "项目数据库使用 openGauss。", 0.9),
        SearchResult("b", "项目部署平台采用鲲鹏 ARM64。", 0.8),
        SearchResult("c", "今天下午天气很好。", 0.1),
    ]
    lines = [f"MEMORY[{i}] {hit.content}" for i, hit in enumerate(candidates, start=1)]
    manager = ContextBudgetManager(store)
    selection = manager.select(
        query="项目数据库和部署平台是什么？",
        candidates=candidates,
        context_lines=lines,
        token_budget=24,
        prefix="HEADER\n",
        suffix="\nQUESTION: query",
    )

    assert selection.tokens_after <= 24
    assert selection.tokens_before >= selection.tokens_after
    assert selection.savings_tokens >= 0
    assert selection.selected
