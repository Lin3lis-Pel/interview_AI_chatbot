"""Schema contracts. These are what stop a bad model reply reaching the UI."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from interview_ai.domain import Evaluation, FinalReport, InterviewPlan, Question, Scores, Topic


def topics(*weights: int) -> list[Topic]:
    return [Topic(name=f"T{i}", weight=w, rationale="because the ad says so") for i, w in enumerate(weights)]


def test_weights_must_total_100():
    InterviewPlan(topics=topics(40, 30, 30))
    with pytest.raises(ValidationError, match="total exactly 100"):
        InterviewPlan(topics=topics(40, 30, 20))
    with pytest.raises(ValidationError, match="total exactly 100"):
        InterviewPlan(topics=topics(50, 30, 30))


def test_plan_needs_at_least_three_topics():
    with pytest.raises(ValidationError):
        InterviewPlan(topics=topics(60, 40))


@pytest.mark.parametrize("n", [5, 8, 10, 13, 16])
def test_question_allocation_totals_exactly_n(n):
    plan = InterviewPlan(topics=topics(35, 30, 20, 15))
    assert sum(plan.question_allocation(n).values()) == n


def test_allocation_favours_heavier_topics():
    plan = InterviewPlan(topics=topics(50, 25, 15, 10))
    alloc = plan.question_allocation(10)
    assert alloc["T0"] >= alloc["T1"] >= alloc["T2"]


def test_scores_are_bounded_and_average():
    s = Scores(correctness=5, depth=4, practical_experience=3, communication=4, relevance=5)
    assert s.mean == 4.2
    with pytest.raises(ValidationError):
        Scores(correctness=6, depth=4, practical_experience=3, communication=4, relevance=5)


def test_non_advance_action_requires_a_follow_up():
    with pytest.raises(ValidationError, match="non-empty follow_up"):
        Evaluation(
            verdict="partial",
            scores=Scores(correctness=3, depth=3, practical_experience=3, communication=3, relevance=3),
            next_action="drill_deeper",
            follow_up="",
        )


def test_advance_may_have_no_follow_up():
    e = Evaluation(
        verdict="strong",
        scores=Scores(correctness=5, depth=5, practical_experience=5, communication=5, relevance=5),
        next_action="advance",
    )
    assert e.follow_up == ""


def test_stub_questions_are_rejected():
    with pytest.raises(ValidationError, match="too short"):
        Question(text="Docker?")


def test_criticality_helpers(analysis):
    assert "Python" in analysis.required_skills
    assert analysis.preferred_skills == ["SQL"]


# --------------------------------------------------------------------------- final report
def report(**lists: list[str]) -> FinalReport:
    return FinalReport(overall_score=50, recommendation="borderline",
                       recommendation_reason="Mixed.", **lists)


def test_report_allows_as_many_unmet_requirements_as_an_ad_can_have():
    """A short interview against a 12-requirement ad must not lose the whole report."""
    unmet = [f"req {i}" for i in range(12)]
    assert report(unmet_requirements=unmet).unmet_requirements == unmet


def test_report_trims_surplus_list_items_instead_of_failing():
    weaknesses = [f"weakness {i}" for i in range(9)]
    r = report(weaknesses=weaknesses, study_plan=[f"step {i}" for i in range(11)])
    assert r.weaknesses == weaknesses[:6]
    assert len(r.study_plan) == 8
