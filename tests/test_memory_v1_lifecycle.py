from datetime import datetime, timedelta, timezone

from memory import ForgettingPolicy, InMemoryMemoryStore, MemoryCompressor, MemoryRecord, MemoryStatus, RecencyScorer


def test_recency_forgetting_and_compression_v1():
    now = datetime.now(timezone.utc)
    scorer = RecencyScorer(decay_rate_per_day=0.1)
    assert scorer.score(now, now=now) > scorer.score(now - timedelta(days=30), now=now)

    store = InMemoryMemoryStore()
    weak = MemoryRecord(
        memory_id="weak",
        content="临时无关记录",
        importance=0.0,
        utility=0.0,
        created_at=now - timedelta(days=365),
    )
    store.add(weak)
    archived = ForgettingPolicy(store, threshold=0.25, recency_scorer=RecencyScorer(0.1)).run(now=now)
    assert archived == ["weak"]
    assert store.get("weak").status == MemoryStatus.ARCHIVED

    long_text = "第一句说明项目背景。第二句说明项目数据库为openGauss。第三句补充大量无关细节。" * 20
    result = MemoryCompressor(max_chars=60).compress(long_text)
    assert result.compressed
    assert result.compressed_chars <= 60
