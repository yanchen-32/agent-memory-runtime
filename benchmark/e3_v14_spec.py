"""Locked semantic surfaces for the E3 v1.4 orthogonal design."""

from __future__ import annotations

from memory import DEFAULT_PREDICATE_ALIASES

from .generate_e3_v13 import PREDICATE_SPECS as _PREDICATE_SPECS
from .generate_e3_v13 import STRATA as _STRATA


DESIGN_VERSION = "v1.4-e3-design.1"
PREDICATE_SPECS = _PREDICATE_SPECS
STRATA = _STRATA
TARGET_POSITION_SPECS = (
    ("front", 1),
    ("middle", 8),
    ("back", 15),
)
QUERY_TEMPLATE_SPECS = (
    {"mode": "canonical_literal", "surface_slot": 0, "template": "{subject}当前的{surface}是什么？"},
    {"mode": "canonical_literal", "surface_slot": 0, "template": "按当前有效状态回答：{subject}的{surface}是什么？"},
    {"mode": "predicate_alias", "surface_slot": 1, "template": "请给出{subject}现行的{surface}？"},
    {"mode": "predicate_alias", "surface_slot": 2, "template": "最新记录中，{subject}采用的{surface}是什么？"},
    {"mode": "nonliteral", "surface_slot": 3, "template": "就{subject}而言，{surface}？"},
    {"mode": "nonliteral", "surface_slot": 4, "template": "查看长期记录后回答：{subject}{surface}？"},
    {"mode": "temporal_alias", "surface_slot": 1, "template": "在查询时点，{subject}的{surface}是哪一项？"},
    {"mode": "temporal_nonliteral", "surface_slot": 3, "template": "忽略已经失效的旧值，{subject}{surface}？"},
)


def predicate_surfaces(predicate: str) -> tuple[str, ...]:
    return (predicate, *DEFAULT_PREDICATE_ALIASES[predicate])


def render_query(subject: str, predicate: str, template_index: int) -> tuple[str, str, str]:
    spec = QUERY_TEMPLATE_SPECS[template_index]
    surface = predicate_surfaces(predicate)[int(spec["surface_slot"])]
    return (
        str(spec["template"]).format(subject=subject, surface=surface),
        str(spec["mode"]),
        surface,
    )
