import pytest

from benchmark import (
    QUANTITY_ANSWER_SCORER_VERSION,
    answer_metrics,
    normalize_answer,
    retrieval_metrics,
)


QUANTITY_SPEC = {
    "type": "quantity",
    "canonical_value": "5",
    "units": ["条"],
    "unit_policy": "optional",
    "output_format": "bare_value",
    "scorer_version": QUANTITY_ANSWER_SCORER_VERSION,
}


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


@pytest.mark.parametrize(
    ("prediction", "format_compliance"),
    [
        ("5", 1),
        ("5。", 1),
        ("5条", 0),
        ("5 条。", 0),
        ("答案是：5条。", 0),
    ],
)
def test_quantity_semantics_are_separate_from_shortest_format(
    prediction,
    format_compliance,
):
    metrics = answer_metrics(prediction, "5", answer_spec=QUANTITY_SPEC)
    assert metrics["semantic_answer_accuracy"] == 1
    assert metrics["answer_accuracy"] == 1
    assert metrics["answer_format_compliance"] == format_compliance
    assert metrics["answer_scorer_version"] == QUANTITY_ANSWER_SCORER_VERSION


@pytest.mark.parametrize(
    "prediction",
    ["大约5条", "5或6", "Top-K为5", "答案是5条，因为配置如此", "6条", "UNKNOWN"],
)
def test_quantity_semantics_reject_ambiguous_or_verbose_predictions(prediction):
    metrics = answer_metrics(prediction, "5", answer_spec=QUANTITY_SPEC)
    assert metrics["semantic_answer_accuracy"] == 0
    assert metrics["answer_format_compliance"] == 0


def test_quantity_semantics_do_not_retroactively_change_legacy_scoring():
    legacy = answer_metrics("5条", "5")
    typed = answer_metrics("5条", "5", answer_spec=QUANTITY_SPEC)
    assert legacy["answer_accuracy"] == 0
    assert typed["strict_answer_accuracy"] == 0
    assert typed["semantic_answer_accuracy"] == 1


def test_quantity_answer_spec_rejects_undeclared_scorer_versions():
    invalid = {**QUANTITY_SPEC, "scorer_version": "future-scorer"}
    with pytest.raises(ValueError, match="scorer_version"):
        answer_metrics("5", "5", answer_spec=invalid)


def test_recall_at_k_requires_all_relevant_memories_for_full_recall():
    metrics = retrieval_metrics(["a", "noise"], ["a", "b"], ks=(1, 2))
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 0.5
