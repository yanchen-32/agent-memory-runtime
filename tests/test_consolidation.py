from memory import InMemoryMemoryStore, MemoryRecord, MemoryType
from memory.consolidation import AdaptiveConsolidationPolicy, MemoryConsolidator
from agent import RuleBasedClient
from benchmark import load_jsonl
from benchmark.runner import run_case
from memory import HashEmbeddingModel
from pathlib import Path


def _episodic(memory_id: str, content: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id="u1",
        memory_type=MemoryType.EPISODIC,
        content=content,
        subject="项目数据库",
        predicate="使用",
        object_value="openGauss",
        importance=0.8,
        confidence=0.9,
    )


def test_consolidation_creates_traceable_semantic_memory_and_is_idempotent():
    store = InMemoryMemoryStore()
    store.add(_episodic("episode-1", "项目数据库使用 openGauss。"))
    store.add(_episodic("episode-2", "项目数据库后来仍使用 openGauss。"))
    engine = MemoryConsolidator(store)

    first = engine.consolidate(user_id="u1")
    second = engine.consolidate(user_id="u1")
    semantic = [record for record in store.list_all("u1") if record.memory_type == MemoryType.SEMANTIC]

    assert len(first.created_ids) == 1
    assert second.created_ids == []
    assert len(second.updated_ids) == 1
    assert len(semantic) == 1
    assert semantic[0].source_ids == ["episode-1", "episode-2"]
    assert semantic[0].metadata["consolidation_engine"] == "episodic-to-semantic-v1"
    assert first.fidelity == 1.0
    assert all(record.status.value == "active" for record in store.list_all("u1"))


def test_consolidation_does_not_merge_conflicting_object_values():
    store = InMemoryMemoryStore()
    store.add(_episodic("episode-old", "项目数据库使用 SQLite。"))
    store.get("episode-old").object_value = "SQLite"
    store.add(_episodic("episode-new", "项目数据库使用 openGauss。"))
    report = MemoryConsolidator(store).consolidate(user_id="u1")

    assert report.created_ids == []
    assert report.groups == []
    assert report.skipped_groups == 2


def test_adaptive_consolidation_records_policy_decision_and_lineage():
    store = InMemoryMemoryStore()
    first = _episodic("episode-1", "项目数据库使用 openGauss。")
    second = _episodic("episode-2", "项目数据库后来仍使用 openGauss。")
    first.access_count = second.access_count = 10
    store.add(first)
    store.add(second)
    policy = AdaptiveConsolidationPolicy(
        trigger_threshold=0.0,
        fine_threshold=0.0,
    )

    report = MemoryConsolidator(store, policy=policy).consolidate(user_id="u1")
    semantic = store.get(report.created_ids[0])

    assert report.groups[0].granularity_level == "fine"
    assert report.groups[0].policy_version == "adaptive-rule-v1"
    assert report.groups[0].trigger_score is not None
    assert semantic.metadata["consolidation_policy_version"] == "adaptive-rule-v1"
    assert semantic.metadata["granularity_level"] == "fine"
    assert semantic.metadata["source_count"] == 2
    assert semantic.metadata["tokens_before"] > 0
    assert semantic.metadata["tokens_after"] > 0


def test_adaptive_consolidation_defers_unresolved_active_conflicts():
    store = InMemoryMemoryStore()
    for memory_id, value in (
        ("sqlite-1", "SQLite"),
        ("sqlite-2", "SQLite"),
        ("gauss-1", "openGauss"),
        ("gauss-2", "openGauss"),
    ):
        record = _episodic(memory_id, f"项目数据库使用 {value}。")
        record.object_value = value
        store.add(record)

    policy = AdaptiveConsolidationPolicy(trigger_threshold=0.0)
    report = MemoryConsolidator(store, policy=policy).consolidate(user_id="u1")

    assert report.created_ids == []
    assert report.conflict_blocked_groups == 2
    assert report.skipped_groups == 2


def test_adaptive_consolidation_can_wait_for_more_evidence():
    store = InMemoryMemoryStore()
    store.add(_episodic("episode-1", "项目数据库使用 openGauss。"))
    store.add(_episodic("episode-2", "项目数据库后来仍使用 openGauss。"))
    policy = AdaptiveConsolidationPolicy(trigger_threshold=2.0)

    report = MemoryConsolidator(store, policy=policy).consolidate(user_id="u1")

    assert report.created_ids == []
    assert report.skipped_by_policy == 1


def test_consolidation_benchmark_really_uses_episodic_sources():
    root = Path(__file__).resolve().parents[1]
    case = load_jsonl(root / "benchmark" / "data" / "consolidation_v0.1.jsonl")[0]
    row = run_case(
        "Ours",
        case,
        RuleBasedClient,
        lambda: HashEmbeddingModel(dim=64),
        consolidation_strategy="adaptive",
        consolidation_policy=AdaptiveConsolidationPolicy(trigger_threshold=0.0),
    )
    assert row["consolidation_source_count"] == 2
    assert row["consolidation_key_fact_recall"] == 1.0
    assert row["consolidation_compression_ratio"] is not None
