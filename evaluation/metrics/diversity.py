"""Is the interview well-shaped, or six versions of one question?

Syllabus structure (weights, subject count, one genuine behavioural subject) and question
spread (type variety, near-duplicates, whether the difficulty dial does anything).
"""
from __future__ import annotations

import re

from interview_ai.domain import InterviewPlan, Question

from ._shared import tokens

_HARD_MARKERS = re.compile(
    r"\b(trade[- ]?off|failure|fail|scale|scaling|bottleneck|latency|cost|why|degrade|"
    r"under load|edge case|regress|contention|consistency|constraint|p9\d)\b", re.IGNORECASE
)


def weights_valid(plan: InterviewPlan) -> float:
    return float(sum(t.weight for t in plan.topics) == 100)


def topic_count_in_range(plan: InterviewPlan) -> float:
    return float(4 <= len(plan.topics) <= 7)


def behavioral_topic_balanced(plan: InterviewPlan) -> float:
    """Exactly one behavioural subject, weighted 5-15%, as the planner prompt requires."""
    rx = re.compile(r"behaviou?r|communicat|collaborat|team|soft skill|culture", re.IGNORECASE)
    matches = [t for t in plan.topics if rx.search(t.name)]
    return float(len(matches) == 1 and 5 <= matches[0].weight <= 15)


def question_type_diversity(questions: list[Question]) -> float:
    """Distinct question types / questions, capped at 1. Flat 1.0 means every one differs."""
    if not questions:
        return 0.0
    return len({q.type for q in questions}) / min(len(questions), 7)


def duplicate_rate(questions: list[Question], threshold: float = 0.6) -> float:
    """Fraction of question pairs that are near-duplicates by token Jaccard. Lower is better."""
    if len(questions) < 2:
        return 0.0
    toks = [tokens(q.text) for q in questions]
    pairs = dups = 0
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            pairs += 1
            union = toks[i] | toks[j]
            if union and len(toks[i] & toks[j]) / len(union) >= threshold:
                dups += 1
    return dups / pairs if pairs else 0.0


def difficulty_separation(easy: list[Question], hard: list[Question]) -> float:
    """Directional: hard questions should carry more trade-off/failure/scale language.

    Returns the marker-rate delta clipped to [0, 1]; 0 means the difficulty knob did nothing.
    """
    def rate(qs: list[Question]) -> float:
        return sum(bool(_HARD_MARKERS.search(q.text)) for q in qs) / len(qs) if qs else 0.0
    return max(0.0, min(1.0, rate(hard) - rate(easy)))
