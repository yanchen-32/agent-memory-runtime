"""Pre-register a full-factorial orthogonal design for a later E3 v1.4."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

from .artifacts import sha256_file
from .generate_e3_v13 import DEFAULT_SEED
from .e3_v14_spec import (
    DESIGN_VERSION,
    PREDICATE_SPECS,
    QUERY_TEMPLATE_SPECS,
    STRATA,
    TARGET_POSITION_SPECS,
)


def build_orthogonal_design() -> list[dict]:
    rows = []
    for index, (predicate_index, query_template_index, target_position_index) in enumerate(
        itertools.product(
            range(len(PREDICATE_SPECS)),
            range(len(QUERY_TEMPLATE_SPECS)),
            range(len(TARGET_POSITION_SPECS)),
        ),
        start=1,
    ):
        rows.append({
            "scenario_family": f"v14_e3_dev_family_{index:03d}",
            "predicate_index": predicate_index,
            "predicate": PREDICATE_SPECS[predicate_index][0],
            "query_template_index": query_template_index,
            "query_expression_mode": QUERY_TEMPLATE_SPECS[query_template_index]["mode"],
            "target_position_index": target_position_index,
            "target_position_band": TARGET_POSITION_SPECS[target_position_index][0],
            "scenario_seed": DEFAULT_SEED + (index - 1) * 10_007,
        })
    return rows


def validate_orthogonal_design(rows: list[dict]) -> dict:
    expected = set(itertools.product(
        range(len(PREDICATE_SPECS)),
        range(len(QUERY_TEMPLATE_SPECS)),
        range(len(TARGET_POSITION_SPECS)),
    ))
    observed = [
        (
            int(row["predicate_index"]),
            int(row["query_template_index"]),
            int(row["target_position_index"]),
        )
        for row in rows
    ]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("v1.4 E3 design must contain every factorial cell exactly once")
    if len(observed) != len(set(observed)):
        raise ValueError("v1.4 E3 design contains duplicate factorial cells")
    predicates = Counter(item[0] for item in observed)
    templates = Counter(item[1] for item in observed)
    positions = Counter(item[2] for item in observed)
    predicate_template = Counter(item[:2] for item in observed)
    predicate_position = Counter((item[0], item[2]) for item in observed)
    template_position = Counter(item[1:] for item in observed)
    if set(predicates.values()) != {len(QUERY_TEMPLATE_SPECS) * len(TARGET_POSITION_SPECS)}:
        raise ValueError("predicate marginal is not balanced")
    if set(templates.values()) != {len(PREDICATE_SPECS) * len(TARGET_POSITION_SPECS)}:
        raise ValueError("query-template marginal is not balanced")
    if set(positions.values()) != {len(PREDICATE_SPECS) * len(QUERY_TEMPLATE_SPECS)}:
        raise ValueError("target-position marginal is not balanced")
    if set(predicate_template.values()) != {len(TARGET_POSITION_SPECS)}:
        raise ValueError("predicate/query pair is not balanced across positions")
    if set(predicate_position.values()) != {len(QUERY_TEMPLATE_SPECS)}:
        raise ValueError("predicate/position pair is not balanced across templates")
    if set(template_position.values()) != {len(PREDICATE_SPECS)}:
        raise ValueError("query/position pair is not balanced across predicates")
    return {
        "valid": True,
        "design_version": DESIGN_VERSION,
        "scenario_families": len(rows),
        "planned_strata": dict(STRATA),
        "planned_development_cases": len(rows) * len(STRATA),
        "factor_levels": {
            "predicates": len(PREDICATE_SPECS),
            "query_templates": len(QUERY_TEMPLATE_SPECS),
            "query_expression_modes": sorted({spec["mode"] for spec in QUERY_TEMPLATE_SPECS}),
            "target_positions": len(TARGET_POSITION_SPECS),
        },
        "full_factorial": True,
        "pairwise_orthogonal": True,
        "test_generated": False,
        "holdout_generated": False,
    }


def write_orthogonal_design(output_dir: str | Path) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = build_orthogonal_design()
    validation = validate_orthogonal_design(rows)
    design_path = root / "orthogonal_design.jsonl"
    with design_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        **validation,
        "status": "design_only_pending_candidate_generation",
        "design_file": design_path.name,
        "design_sha256": sha256_file(design_path),
        "source_benchmark": "v1.3-e3 Development diagnostics only",
        "pre_registration_note": (
            "Semantic generation is authorized for the locked memory-runtime-v2 "
            "method; Test remains ungenerated until Development gates pass."
        ),
    }
    (root / "design_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the orthogonal E3 v1.4 design.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data/v1.4-e3",
    )
    args = parser.parse_args()
    print(json.dumps(write_orthogonal_design(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
