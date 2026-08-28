from memory import (
    HashEmbeddingModel,
    ImportanceScorer,
    MemoryCandidate,
    MemoryType,
    RuleMemoryClassifier,
    RuleMemoryExtractor,
)


def test_extraction_classification_importance_v1():
    extractor = RuleMemoryExtractor()
    candidates = extractor.extract([{"role": "user", "content": "Agent Memory项目的部署平台确定为鲲鹏。"}])
    assert len(candidates) == 1
    c = candidates[0]
    assert c.predicate == "部署平台"
    assert "鲲鹏" in c.object_value
    memory_type, confidence = RuleMemoryClassifier().classify(c)
    assert memory_type == MemoryType.SEMANTIC
    assert confidence >= 0.8
    assert ImportanceScorer().score(c) > 0.4


def test_importance_plain_chitchat_is_lower():
    scorer = ImportanceScorer()
    important = MemoryCandidate(content="请记住项目最终截止日期是9月15日", subject="项目", predicate="截止日期")
    casual = MemoryCandidate(content="今天食堂的面一般")
    assert scorer.score(important) > scorer.score(casual)
