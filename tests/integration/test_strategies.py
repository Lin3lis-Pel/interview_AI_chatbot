"""Prompting-strategy machinery, exemplar hygiene, token caps and cross-judge metrics.

The most important test here is `test_rules_are_identical_across_strategies`: if the arms of
the ablation differ in more than the appended block, the matrix measures nothing.
"""
from __future__ import annotations

import pytest

from interview_ai.config import SETTINGS
from interview_ai.prompts import (
    EXEMPLAR_FINGERPRINTS,
    PRODUCT_STAGES,
    STAGES,
    STRATEGIES,
    build_system,
)
from interview_ai.domain import (
    Evaluation,
    InterviewPlan,
    JobAnalysis,
    JudgeAnswerScore,
    JudgeQuestionRating,
    Requirement,
    Scores,
    Topic,
)
from evaluation import metrics as M
from evaluation.fixtures.job_ads import ALL_ADS
from evaluation.runners.benchmark import composite, shared_metric_names
from tests.conftest import make_evaluation


# --------------------------------------------------------------------------- assembly
@pytest.mark.parametrize("stage", PRODUCT_STAGES)
def test_rules_are_identical_across_strategies(stage):
    """Every variant must start with byte-identical rules -- otherwise the ablation is invalid."""
    fmt = {"n": 3} if stage == "questioner" else {"max_followups": 2} if stage == "evaluator" else {}
    base = build_system(stage, "zero_shot", **fmt)
    for strategy in STRATEGIES:
        assert build_system(stage, strategy, **fmt).startswith(base), (
            f"{stage}/{strategy} altered the shared rules"
        )


@pytest.mark.parametrize("stage", PRODUCT_STAGES)
def test_exemplar_counts(stage):
    fmt = {"n": 3} if stage == "questioner" else {"max_followups": 2} if stage == "evaluator" else {}
    available = len(STAGES[stage].exemplars)
    one = build_system(stage, "one_shot", **fmt)
    few = build_system(stage, "few_shot", **fmt)
    assert one.count("--- EXAMPLE ") == 1
    assert few.count("--- EXAMPLE ") == min(3, available)
    assert len(few) >= len(one)


@pytest.mark.parametrize("stage", PRODUCT_STAGES)
def test_guided_adds_a_procedure_and_no_examples(stage):
    fmt = {"n": 3} if stage == "questioner" else {"max_followups": 2} if stage == "evaluator" else {}
    guided = build_system(stage, "guided", **fmt)
    assert "PROCEDURE" in guided and "Step 1." in guided
    assert "--- EXAMPLE " not in guided


@pytest.mark.parametrize("stage", PRODUCT_STAGES)
def test_self_critique_adds_a_stage_checklist_and_no_examples(stage):
    fmt = {"n": 3} if stage == "questioner" else {"max_followups": 2} if stage == "evaluator" else {}
    prompt = build_system(stage, "self_critique", **fmt)
    assert "SELF-CRITIQUE CHECKLIST" in prompt
    assert "audit" in prompt.lower() and "Return only the final JSON" in prompt
    assert "--- EXAMPLE " not in prompt and "PROCEDURE" not in prompt


def test_exactly_five_comparable_product_strategies():
    assert STRATEGIES == (
        "zero_shot", "one_shot", "few_shot", "guided", "self_critique",
    )
    assert len(PRODUCT_STAGES) == 5


@pytest.mark.parametrize("stage", PRODUCT_STAGES)
def test_zero_shot_is_rules_only(stage):
    fmt = {"n": 3} if stage == "questioner" else {"max_followups": 2} if stage == "evaluator" else {}
    z = build_system(stage, "zero_shot", **fmt)
    assert "--- EXAMPLE " not in z and "PROCEDURE" not in z
    assert "SELF-CRITIQUE CHECKLIST" not in z


def test_exemplar_blocks_warn_against_copying():
    few = build_system("planner", "few_shot")
    assert "SHAPE REFERENCE ONLY" in few
    assert "Do NOT copy" in few
    assert "DIFFERENT job" in few


def test_format_placeholders_survive_exemplar_json_braces():
    """Exemplar blocks contain JSON braces; they must never reach str.format."""
    for strategy in STRATEGIES:
        q = build_system("questioner", strategy, n=4)
        assert "Write 4 questions" in q
        assert "{n}" not in q
        e = build_system("evaluator", strategy, max_followups=2)
        assert "spent\n                    2 follow-ups" in e or "2 follow-ups" in e
        assert "{max_followups}" not in e


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_system("analyzer", "chain_of_thought")


def test_judge_stages_have_no_exemplars_but_do_have_a_guide():
    """The second opinion must stay independent: a guided procedure is fine (it carries no
    content from the primary prompts), but a shared worked example would correlate the two
    scorers, so exemplar strategies fall back to rules only."""
    for stage in ("judge_answer", "judge_question"):
        assert STAGES[stage].exemplars == []
        base = build_system(stage, "zero_shot")
        assert build_system(stage, "few_shot") == base
        assert build_system(stage, "one_shot") == base
        assert build_system(stage, "self_critique") == base
        assert build_system(stage, "guided") != base
        assert "PROCEDURE" in build_system(stage, "guided")


@pytest.mark.parametrize("stage", PRODUCT_STAGES)
def test_every_product_stage_has_all_strategy_material(stage):
    assert STAGES[stage].guide.strip(), f"{stage} has no guided variant"
    assert STAGES[stage].critique.strip(), f"{stage} has no self-critique variant"
    assert STAGES[stage].exemplars, f"{stage} has no worked example"


def test_exemplar_roles_do_not_collide_with_eval_fixtures():
    """Contamination guard: if an exemplar shared a role with a fixture, leakage would be
    indistinguishable from a correct answer."""
    fixture_roles = {ad.role for ad in ALL_ADS}
    assert not (set(EXEMPLAR_FINGERPRINTS["roles"]) & fixture_roles)


# --------------------------------------------------------------------------- token caps
@pytest.mark.parametrize("stage", PRODUCT_STAGES)
def test_every_stage_has_an_output_token_cap(stage):
    assert SETTINGS.max_tokens_for(stage) > 0


def test_reporter_gets_the_largest_budget():
    """The report is the longest output; a cap tuned for questions would truncate it."""
    assert SETTINGS.max_tokens_for("reporter") > SETTINGS.max_tokens_for("questioner")


def test_per_stage_override_and_fallback(monkeypatch):
    assert SETTINGS.strategy_for("analyzer") == SETTINGS.strategy
    monkeypatch.setenv("INTERVIEW_STRATEGY_ANALYZER", "few_shot")
    assert SETTINGS.strategy_for("analyzer") == "few_shot"
    assert SETTINGS.strategy_for("planner") == SETTINGS.strategy


# --------------------------------------------------------------------------- exemplar leakage
def clean_plan() -> InterviewPlan:
    return InterviewPlan(topics=[
        Topic(name="Retrieval & RAG", weight=40, rationale="RAG is the core of the ad",
              grounded_in=["RAG"]),
        Topic(name="Python services", weight=32, rationale="Python is non-negotiable",
              grounded_in=["Python"]),
        Topic(name="Containers & GCP", weight=18, rationale="Docker and GCP listed",
              grounded_in=["Docker"]),
        Topic(name="Team communication", weight=10, rationale="hybrid team",
              grounded_in=["Python"]),
    ])


def test_clean_plan_has_no_leakage():
    assert M.plan_exemplar_leakage(clean_plan()) == 1.0


def test_copied_exemplar_subject_names_are_caught():
    copied = InterviewPlan(topics=[
        Topic(name="Flaky test diagnosis", weight=40, rationale="x", grounded_in=["RAG"]),
        Topic(name="Retrieval & RAG", weight=32, rationale="x", grounded_in=["RAG"]),
        Topic(name="Python services", weight=18, rationale="x", grounded_in=["Python"]),
        Topic(name="Team communication", weight=10, rationale="x", grounded_in=["Python"]),
    ])
    assert M.plan_exemplar_leakage(copied) < 1.0


def shaped_plan(first_name: str) -> InterviewPlan:
    """A plan whose weights match the exemplars' 30/25/20/15/10 multiset."""
    return InterviewPlan(topics=[
        Topic(name=first_name, weight=30, rationale="x", grounded_in=["RAG"]),
        Topic(name="Python services", weight=25, rationale="x", grounded_in=["Python"]),
        Topic(name="Containers", weight=20, rationale="x", grounded_in=["Docker"]),
        Topic(name="Cloud on GCP", weight=15, rationale="x", grounded_in=["GCP"]),
        Topic(name="Team communication", weight=10, rationale="x", grounded_in=["Python"]),
    ])


def test_weight_shape_alone_does_not_accuse():
    """Regression: the first full matrix fired this on zero_shot, whose prompt has no
    examples to copy. A smooth descending split is just what sane weighting looks like."""
    assert M.plan_exemplar_leakage(shaped_plan("Retrieval & RAG")) == 1.0


def test_weight_shape_corroborates_a_real_hit():
    """With a copied subject name present, the matching shape adds to the evidence."""
    with_name = M.plan_exemplar_leakage(shaped_plan("Flaky test diagnosis"))
    without_shape = M.plan_exemplar_leakage(InterviewPlan(topics=[
        Topic(name="Flaky test diagnosis", weight=40, rationale="x", grounded_in=["RAG"]),
        Topic(name="Python services", weight=32, rationale="x", grounded_in=["Python"]),
        Topic(name="Containers", weight=18, rationale="x", grounded_in=["Docker"]),
        Topic(name="Team communication", weight=10, rationale="x", grounded_in=["Python"]),
    ]))
    assert with_name < without_shape < 1.0


def test_terms_present_in_the_ad_are_not_leakage():
    """Regression: the frontend_mid fixture genuinely lists Playwright, so echoing it is
    correct behaviour. Flagging it made the metric fire on every arm, zero_shot included."""
    from evaluation.fixtures.job_ads import FRONTEND

    plan = InterviewPlan(topics=[
        Topic(name="End-to-end testing", weight=40,
              rationale="The ad requires Playwright and Jest tests", grounded_in=["testing"]),
        Topic(name="React components", weight=30, rationale="x", grounded_in=["react"]),
        Topic(name="Accessibility", weight=20, rationale="x", grounded_in=["accessibility"]),
        Topic(name="Team communication", weight=10, rationale="x", grounded_in=["react"]),
    ])
    assert M.plan_exemplar_leakage(plan, FRONTEND.text) == 1.0
    assert M.plan_exemplar_leakage(plan) < 1.0  # unfiltered, the old false positive


def test_analysis_leakage_detects_exemplar_role_and_terms(analysis):
    assert M.analysis_exemplar_leakage(analysis, "AI Engineer") == 1.0
    wrong_role = analysis.model_copy(update={"role": "QA / Automation"})
    assert M.analysis_exemplar_leakage(wrong_role, "AI Engineer") < 1.0
    with_terms = JobAnalysis(
        is_job_ad=True, role="AI Engineer", technologies=["Playwright", "k6"],
        requirements=[Requirement(name="RAG", criticality="critical", evidence="RAG")],
    )
    assert M.analysis_exemplar_leakage(with_terms, "AI Engineer") < 1.0


def test_question_leakage(questions):
    from interview_ai.domain import Question

    assert M.question_exemplar_leakage(questions) == 1.0
    leaked = questions + [Question(
        text="How would you debug a flaky test in your Playwright suite for this role?",
        targets=["Python"])]
    assert M.question_exemplar_leakage(leaked) < 1.0


# --------------------------------------------------------------------------- cross-judge
def judge_score(mean_parts: int, verdict: str = "partial") -> JudgeAnswerScore:
    return JudgeAnswerScore(
        verdict=verdict,
        scores=Scores(correctness=mean_parts, depth=mean_parts, practical_experience=mean_parts,
                      communication=mean_parts, relevance=mean_parts),
        reasoning="because",
    )


def test_score_agreement_within_tolerance():
    primary = make_evaluation(score=4, verdict="partial")
    assert M.judge_score_agreement(primary, judge_score(4)) == 1.0
    assert M.judge_score_agreement(primary, judge_score(3)) == 1.0   # within 1.0
    assert M.judge_score_agreement(primary, judge_score(2)) == 0.0   # two steps apart


def test_verdict_agreement_is_graded():
    primary = make_evaluation(verdict="vague")
    assert M.judge_verdict_agreement(primary, judge_score(2, "vague")) == 1.0
    assert M.judge_verdict_agreement(primary, judge_score(2, "partial")) == 0.5   # adjacent
    assert M.judge_verdict_agreement(primary, judge_score(5, "strong")) == 0.0


def test_bias_sign_reveals_self_preference():
    """Positive bias = the primary model rated its own generation above the judge."""
    generous = make_evaluation(score=5, verdict="strong")
    assert M.judge_score_bias(generous, judge_score(2)) == pytest.approx(3.0)
    harsh = make_evaluation(score=2, verdict="vague")
    assert M.judge_score_bias(harsh, judge_score(4)) == pytest.approx(-2.0)
    assert M.judge_score_bias(generous, judge_score(5)) == 0.0


def test_judged_question_quality_and_fit():
    good = JudgeQuestionRating(grounded=True, realistic=True, single_ask=True,
                               difficulty_fit="about_right")
    compound = JudgeQuestionRating(grounded=True, realistic=True, single_ask=False,
                                   difficulty_fit="too_easy")
    assert good.quality == 1.0
    assert compound.quality == pytest.approx(2 / 3)
    assert M.judged_question_quality([good, compound]) == pytest.approx(5 / 6)
    assert M.judged_difficulty_fit([good, compound]) == 0.5
    assert M.judged_question_quality([]) == 0.0


# --------------------------------------------------------------------------- matrix maths
def arm(name: str, metrics: dict[str, float], cost: float = 0.01) -> dict:
    return {
        "strategy": name,
        "metrics": {k: {"mean": v, "n": 1, "min": v} for k, v in metrics.items()},
        "usage": {"cost_usd": cost, "total_tokens": 1000, "p50_latency_s": 1.0},
        "n_passed": 0, "n_thresholded": 0, "passed": False, "failures": [],
    }


def test_shared_metric_names_only_keeps_common_thresholded_metrics():
    a = arm("zero_shot", {"role_accuracy": 1.0, "evidence_grounding": 0.9})
    b = arm("few_shot", {"role_accuracy": 0.8, "schema_validity": 1.0})
    assert shared_metric_names([a, b]) == ["role_accuracy"]


def test_composite_corrects_orientation_for_lower_is_better():
    good = arm("a", {"question_duplicate_rate": 0.0, "role_accuracy": 1.0})
    bad = arm("b", {"question_duplicate_rate": 1.0, "role_accuracy": 1.0})
    names = ["question_duplicate_rate", "role_accuracy"]
    assert composite(good, names) == 1.0
    assert composite(bad, names) == 0.5


def test_composite_ignores_metrics_an_arm_did_not_produce():
    a = arm("a", {"role_accuracy": 1.0})
    assert composite(a, ["role_accuracy", "judge_score_agreement"]) == 1.0


def test_thresholds_exist_for_the_new_metrics():
    for name in ("analysis_exemplar_clean", "plan_exemplar_clean", "question_exemplar_clean",
                 "judge_score_agreement", "judge_verdict_agreement", "judged_question_quality"):
        assert name in M.THRESHOLDS
    # A signed diagnostic must never be pass/failed.
    assert "judge_score_bias" not in M.THRESHOLDS
    assert M.verdict("judge_score_bias", -99.0) is True
