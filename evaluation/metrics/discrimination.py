"""Does the scorer actually separate a good answer from a bad one?

A scorer that rates everything a 4 passes every grounding check and is useless. These
metrics measure spread across the gold tiers, whether the flow-control number agrees with
the verdict, and how far the primary model drifts from an independent judge.
"""
from __future__ import annotations

import re

from interview_ai.domain import Evaluation, JudgeAnswerScore, JudgeQuestionRating

from ._shared import VERDICT_ORDER


def action_matches_policy(evaluation: Evaluation, expected: set[str]) -> float:
    return float(evaluation.next_action in expected)


def score_in_band(evaluation: Evaluation, band: tuple[float, float]) -> float:
    lo, hi = band
    return float(lo <= evaluation.scores.mean <= hi)


def follow_up_is_conversational(evaluation: Evaluation) -> float:
    """One question, no lecture, no verdict leak, reasonable length."""
    fu = evaluation.follow_up.strip()
    if not fu:
        return float(evaluation.next_action == "advance")
    leaks = re.search(r"\b(correct|incorrect|wrong|score|you scored|good answer|well done)\b", fu, re.I)
    return float(len(fu) <= 300 and fu.count("?") <= 1 and not leaks)


def discrimination(mean_by_tier: dict[str, float]) -> float:
    """Are tiers ordered strong > partial > vague > wrong? Fraction of ordered adjacent pairs."""
    order = [t for t in ("strong", "partial", "vague", "wrong", "no_answer") if t in mean_by_tier]
    if len(order) < 2:
        return 0.0
    ok = sum(mean_by_tier[a] > mean_by_tier[b] for a, b in zip(order, order[1:]))
    return ok / (len(order) - 1)


def coverage_discrimination(coverage_by_tier: dict[str, float]) -> float:
    """Does `information_coverage` fall as answer quality falls?

    Coverage now drives the interview loop -- a threshold decides whether to follow up and
    whether the requirement counts as demonstrated -- so it has to track answer quality at
    least as reliably as the 1-5 scores do. If it does not, the interview follows up on the
    wrong answers and the report's unmet-requirements section is noise.
    """
    return discrimination(coverage_by_tier)


def coverage_matches_verdict(evaluation: Evaluation) -> float:
    """Coverage and verdict must tell the same story about the same answer.

    A `strong` verdict paired with 0.2 coverage (or `no_answer` with 0.9) means the model
    filled the fields independently instead of judging one answer, which would make flow
    control disagree with scoring.
    """
    # Bands are deliberately wide, because verdict and coverage measure DIFFERENT things:
    # the verdict grades the answer, coverage grades how much of the rubric is satisfied.
    # A `partial` answer against a demanding rubric can legitimately sit at 0.25 -- the
    # first run of this metric flagged exactly that and it was the band, not the model,
    # that was wrong (2026-08-25: verdict `partial` at coverage 0.25 on two arms, where
    # the gold tier was `vague`, i.e. coverage was the MORE accurate of the two numbers).
    # What must never happen is a flat contradiction: "complete" paired with an empty
    # rubric, or a refusal paired with a satisfied one. Those are what this guards.
    bands = {
        "strong": (0.55, 1.0), "partial": (0.05, 0.9), "vague": (0.0, 0.65),
        "wrong": (0.0, 0.55), "no_answer": (0.0, 0.25),
    }
    lo, hi = bands.get(evaluation.verdict, (0.0, 1.0))
    return float(lo <= evaluation.information_coverage <= hi)


# Same model writing and grading cannot detect its own generosity. An independent judge on a
# different model family can. These metrics measure agreement, not who is right.
def judge_score_agreement(primary: Evaluation, judge: JudgeAnswerScore, tolerance: float = 1.0) -> float:
    """1.0 when the two mean scores are within `tolerance` on the 1-5 scale."""
    return float(abs(primary.scores.mean - judge.scores.mean) <= tolerance)


def judge_verdict_agreement(primary: Evaluation, judge: JudgeAnswerScore) -> float:
    """1.0 for the same verdict, 0.5 for adjacent, 0.0 for two or more steps apart."""
    try:
        gap = abs(VERDICT_ORDER.index(primary.verdict) - VERDICT_ORDER.index(judge.verdict))
    except ValueError:
        return 0.0
    return {0: 1.0, 1: 0.5}.get(gap, 0.0)


def judge_score_bias(primary: Evaluation, judge: JudgeAnswerScore) -> float:
    """Signed: positive means the primary model scored ITS OWN work higher than the judge did.

    Not a pass/fail metric -- reported as a diagnostic. A consistently positive bias across
    tiers is the self-preference signature.
    """
    return round(primary.scores.mean - judge.scores.mean, 3)


def judged_question_quality(ratings: list[JudgeQuestionRating]) -> float:
    """Mean of the grounded / realistic / single-ask checks across rated questions."""
    if not ratings:
        return 0.0
    return sum(r.quality for r in ratings) / len(ratings)


def judged_difficulty_fit(ratings: list[JudgeQuestionRating]) -> float:
    if not ratings:
        return 0.0
    return sum(r.difficulty_fit == "about_right" for r in ratings) / len(ratings)
