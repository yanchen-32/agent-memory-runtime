from benchmark import answer_metrics, normalize_answer, retrieval_metrics


def test_answer_metrics_separate_raw_normalized_accuracy_and_f1():
    wrapped = answer_metrics("答案是 OpenGauss。", "openGauss")
    assert wrapped["exact_match"] is False
    assert wrapped["normalized_match"] is True
    assert wrapped["answer_accuracy"] == 1
    assert wrapped["answer_f1"] == 1.0

    verbose = answer_metrics("项目数据库使用 openGauss", "openGauss")
    assert verbose["normalized_match"] is False
    assert 0.0 < verbose["answer_f1"] < 1.0

    explained = answer_metrics("答案是 openGauss。因为已经完成迁移。", "openGauss")
    assert explained["normalized_match"] is False
    assert explained["answer_match"] is True
    assert explained["answer_accuracy"] == 1


def test_answer_metrics_use_only_frozen_aliases():
    without_alias = answer_metrics("postgresql", "openGauss")
    with_alias = answer_metrics("PostgreSQL。", "openGauss", ["PostgreSQL"])
    assert without_alias["answer_accuracy"] == 0
    assert with_alias["answer_accuracy"] == 1
    assert with_alias["matched_target"] == "PostgreSQL"
    assert normalize_answer("回答： PostgreSQL。") == "postgresql"


def test_recall_at_k_requires_all_relevant_memories_for_full_recall():
    metrics = retrieval_metrics(["a", "noise"], ["a", "b"], ks=(1, 2))
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 0.5
