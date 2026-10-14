"""Independent judges.

The same model writing and grading cannot detect its own generosity. A judge is a second
model on a separate endpoint -- by default a small local one over Ollama -- that re-scores
the same answers, so `judge_score_bias` can show the primary model marking its own work.

Judge prompts get a `guided` procedure but deliberately NO exemplars: a worked example
shared with the primary model would correlate the two scorers, and a correlated second
opinion measures nothing.
"""
from __future__ import annotations

from .gemma import (
    JudgeUnavailable,
    build_judge,
    judge_available,
    rate_question,
    rate_questions,
    score_answer,
)

__all__ = [
    "JudgeUnavailable", "build_judge", "judge_available", "rate_question",
    "rate_questions", "score_answer",
]
