from memory import HashEmbeddingModel, MemoryAction, MemoryRuntimeV1, MemoryStatus


def test_runtime_v1_end_to_end_write_update_read():
    runtime = MemoryRuntimeV1(embedder=HashEmbeddingModel(dim=256))
    first = runtime.write([{"role": "user", "content": "Agent Memory项目的数据库是SQLite。"}], user_id="u1")
    assert first[0].action == MemoryAction.ADD
    second = runtime.write([{"role": "user", "content": "Agent Memory项目的数据库改为openGauss。"}], user_id="u1")
    assert second[0].action == MemoryAction.SUPERSEDE
    all_records = runtime.store.list_all("u1")
    assert len(all_records) == 2
    assert sum(r.status == MemoryStatus.ACTIVE for r in all_records) == 1
    result = runtime.read("Agent Memory项目数据库是什么？", top_k=1, user_id="u1")
    assert result.hits
    assert "openGauss" in result.context
