from datetime import datetime, timezone

from memory import HashEmbeddingModel, MemoryAction, MemoryRuntimeV1


def test_runtime_reads_current_and_historical_versions():
    runtime = MemoryRuntimeV1(embedder=HashEmbeddingModel(dim=128))
    t1 = datetime(2026, 8, 1, 9, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 15, 9, tzinfo=timezone.utc)
    t3 = datetime(2026, 8, 20, 9, tzinfo=timezone.utc)

    first = runtime.write(
        [{"role": "user", "content": "项目数据库是 SQLite。"}],
        user_id="u1",
        at=t1,
    )
    second = runtime.write(
        [{"role": "user", "content": "项目数据库改为 openGauss。"}],
        user_id="u1",
        at=t2,
    )

    assert first[0].action == MemoryAction.ADD
    assert second[0].action == MemoryAction.SUPERSEDE

    current = runtime.read("项目数据库是什么？", top_k=1, user_id="u1", query_time=t3)
    historical = runtime.read("项目数据库是什么？", top_k=1, user_id="u1", query_time=t1)

    assert current.hits
    assert historical.hits
    assert "openGauss" in current.context
    assert "SQLite" in historical.context
    assert current.query_time == t3
    assert historical.query_time == t1
