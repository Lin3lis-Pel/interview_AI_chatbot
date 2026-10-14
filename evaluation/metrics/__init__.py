"""Pure metric functions. No network, no model -- so they are unit-tested offline.

Every metric returns a plain float in [0, 1] (or a count), so the runner can average them
across fixtures and print one table. Grouped by the question they answer:

    grounding.py       is the output tied to the ad, or invented / copied from an exemplar?
    diversity.py       is the interview well-shaped, or six versions of one question?
    discrimination.py  does the scorer separate a good answer from a bad one?
    security.py        did the guard rails hold against the attack fixtures?

Everything is re-exported here, so `from evaluation import metrics as M` then `M.<name>` reaches
any of them regardless of which file it lives in.

The pass bar lives in `evaluation/thresholds.yaml`, not in this code -- edit it there and rerun.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ._shared import SENIORITY_ORDER, VERDICT_ORDER, mentions, norm, tokens
from .discrimination import (
    action_matches_policy,
    coverage_discrimination,
    coverage_matches_verdict,
    discrimination,
    follow_up_is_conversational,
    judge_score_agreement,
    judge_score_bias,
    judge_verdict_agreement,
    judged_difficulty_fit,
    judged_question_quality,
    score_in_band,
)
from .diversity import (
    behavioral_topic_balanced,
    difficulty_separation,
    duplicate_rate,
    question_type_diversity,
    topic_count_in_range,
    weights_valid,
)
from .grounding import (
    analysis_exemplar_leakage,
    criticality_precision,
    emphasis_detected,
    evidence_grounding,
    plan_exemplar_leakage,
    question_exemplar_leakage,
    question_grounding,
    requirement_recall,
    role_correct,
    rubric_present,
    seniority_exact,
    seniority_within_one,
    syllabus_coverage,
    topic_grounding,
)
from .security import no_prompt_leak, resisted_ad_injection, resisted_score_injection


# --------------------------------------------------------------------------- aggregation
@dataclass
class MetricTable:
    rows: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.rows.setdefault(name, []).append(float(value))

    def merge(self, other: "MetricTable") -> None:
        for k, v in other.rows.items():
            self.rows.setdefault(k, []).extend(v)

    def mean(self, name: str) -> float:
        vals = self.rows.get(name, [])
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {"mean": self.mean(name), "n": len(vals), "min": round(min(vals), 3)}
            for name, vals in sorted(self.rows.items())
        }


# --------------------------------------------------------------------------- pass/fail
_THRESHOLD_FILE = Path(__file__).resolve().parents[1] / "thresholds.yaml"
_LOADED = yaml.safe_load(_THRESHOLD_FILE.read_text(encoding="utf-8"))

THRESHOLDS: dict[str, float] = {k: float(v) for k, v in _LOADED["thresholds"].items()}
LOWER_IS_BETTER: set[str] = set(_LOADED.get("lower_is_better") or [])


def verdict(name: str, value: float) -> bool:
    threshold = THRESHOLDS.get(name)
    if threshold is None:
        return True
    return value <= threshold if name in LOWER_IS_BETTER else value >= threshold
