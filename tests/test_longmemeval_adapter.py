import json
from pathlib import Path

from benchmark.adapters.longmemeval import convert_case
from benchmark.validate_v1 import validate_file


def _source_case():
    return {
        "question_id": "sample_1",
        "question_type": "knowledge-update",
        "question": "Which database do I use now?",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer": "openGauss",
        "answer_session_ids": ["session_2"],
        "haystack_dates": [
            "2023/05/20 (Sat) 02:21",
            "2023/05/21 (Sun) 03:24",
        ],
        "haystack_session_ids": ["session_1", "session_2"],
        "haystack_sessions": [
            [{"role": "user", "content": "I use SQLite."}],
            [{"role": "user", "content": "I switched to openGauss."}],
        ],
    }


def test_longmemeval_adapter_preserves_evidence_and_ground_truth():
    converted = convert_case(_source_case())
    assert converted["category"] == "update"
    assert converted["expected_memory_ids"] == ["session_2"]
    assert converted["expected_answer"] == "openGauss"
    assert converted["metadata"]["ground_truth_modified"] is False
    assert converted["conversation"][1]["memory_id"] == "session_2"
    assert "I switched to openGauss." in converted["conversation"][1]["content"]


def test_v1_validator_accepts_converted_case(tmp_path: Path):
    output = tmp_path / "converted.jsonl"
    output.write_text(json.dumps(convert_case(_source_case())) + "\n", encoding="utf-8")
    report = validate_file(output)
    assert report["valid"] is True
    assert report["cases"] == 1


def test_longmemeval_adapter_suffixes_only_duplicate_filler_ids():
    source = _source_case()
    source["haystack_session_ids"] = ["filler", "filler", "answer"]
    source["haystack_dates"] = [
        "2023/05/20 (Sat) 02:21",
        "2023/05/20 (Sat) 03:21",
        "2023/05/21 (Sun) 03:24",
    ]
    source["haystack_sessions"] = [
        [{"role": "user", "content": "Repeated filler."}],
        [{"role": "user", "content": "Repeated filler."}],
        [{"role": "user", "content": "I switched to openGauss."}],
    ]
    source["answer_session_ids"] = ["answer"]
    converted = convert_case(source)
    assert [turn["memory_id"] for turn in converted["conversation"]] == [
        "filler__occ1", "filler__occ2", "answer"
    ]
    assert converted["expected_memory_ids"] == ["answer"]
