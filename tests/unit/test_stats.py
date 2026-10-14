"""Statistics layer and artifact persistence -- both offline."""
from __future__ import annotations

import json
import math

import pytest

from interview_ai.prompts import JUDGE_STAGES, build_system
from interview_ai.domain import JudgeAnswerScore, Scores
from evaluation import stats as S
from evaluation.artifacts import ArtifactLog, load_artifacts, questions_from


# --------------------------------------------------------------------------- judge guides
@pytest.mark.parametrize("stage", JUDGE_STAGES)
def test_judge_has_a_guided_variant_but_no_exemplars(stage):
    base = build_system(stage, "zero_shot")
    guided = build_system(stage, "guided")
    assert guided != base, f"{stage} has no guided procedure"
    assert "PROCEDURE" in guided and "Step 1." in guided
    # Exemplars shared with the primary model would correlate the two scorers.
    assert build_system(stage, "few_shot") == base
    assert "--- EXAMPLE " not in guided


def test_judge_guide_forces_evidence_counting():
    """The step that stops a small model rewarding fluency."""
    guided = build_system("judge_answer", "guided")
    assert "count it" in guided
    assert "every number is 4" in guided


def test_judge_prompt_never_shows_the_primary_score():
    """Showing it would produce anchoring, not an independent second opinion."""
    for strategy in ("zero_shot", "guided"):
        text = build_system("judge_answer", strategy)
        assert "primary" not in text.lower()


# --------------------------------------------------------------------------- agreement
def test_perfect_agreement():
    r = S.rater_agreement([4.0, 2.0, 1.0], [4.0, 2.0, 1.0],
                          ["strong", "vague", "wrong"], ["strong", "vague", "wrong"])
    assert r.n == 3
    assert r.mae == 0.0
    assert r.mean_bias == 0.0
    assert r.cohen_kappa == 1.0
    assert r.spearman_rho == 1.0
    assert r.exact_verdict_match == 1.0


def test_positive_bias_flags_self_preference():
    """Primary consistently above the judge -- the signature we are hunting."""
    r = S.rater_agreement([5.0, 4.0, 4.0], [3.0, 2.0, 2.0],
                          ["strong", "strong", "partial"], ["vague", "wrong", "wrong"])
    assert r.mean_bias == pytest.approx(2.0)
    assert r.mae == pytest.approx(2.0)


def test_negative_bias_is_reported_as_negative():
    r = S.rater_agreement([2.0, 2.0], [4.0, 4.0], ["vague", "vague"], ["strong", "strong"])
    assert r.mean_bias < 0


def test_kappa_punishes_chance_agreement():
    """A judge that answers 'partial' every time agrees often but has no reliability."""
    primary = ["partial", "strong", "wrong", "partial", "vague"]
    lazy = ["partial"] * 5
    r = S.rater_agreement([3.0, 5.0, 1.0, 3.0, 2.0], [3.0] * 5, primary, lazy)
    assert r.exact_verdict_match == pytest.approx(0.4)   # looks respectable
    assert r.cohen_kappa <= 0.0                          # reliability says otherwise


def test_constant_scores_give_nan_not_a_fake_zero():
    r = S.rater_agreement([3.0, 3.0, 3.0], [3.0, 3.0, 3.0],
                          ["partial"] * 3, ["partial"] * 3)
    assert math.isnan(r.spearman_rho)


def test_empty_input_is_safe():
    r = S.rater_agreement([], [], [], [])
    assert r.n == 0 and math.isnan(r.cohen_kappa)


# --------------------------------------------------------------------------- rank correlation
def test_tier_correlation_rewards_a_discriminating_scorer():
    tiers = ["strong", "partial", "vague", "wrong", "no_answer"]
    good = S.tier_rank_correlation(tiers, [4.8, 3.4, 2.2, 1.6, 1.0])
    assert good["spearman_rho"] == 1.0
    inverted = S.tier_rank_correlation(tiers, [1.0, 1.6, 2.2, 3.4, 4.8])
    assert inverted["spearman_rho"] == -1.0


def test_flat_scorer_gets_nan():
    tiers = ["strong", "partial", "vague", "wrong"]
    assert math.isnan(S.tier_rank_correlation(tiers, [4.0] * 4)["spearman_rho"])


# --------------------------------------------------------------------------- classification
def test_role_classification_surfaces_a_systematic_mixup():
    gold = ["ML Engineer", "ML Engineer", "Data Scientist", "AI Engineer"]
    pred = ["Data Scientist", "Data Scientist", "Data Scientist", "AI Engineer"]
    out = S.role_classification(gold, pred)
    assert out["accuracy"] == 0.5
    assert out["per_class"]["ML Engineer"]["recall"] == 0.0
    assert out["per_class"]["Data Scientist"]["precision"] < 1.0
    assert len(out["confusion_matrix"]) == len(out["confusion_index"])
    assert S.render_confusion(out)[0].startswith("| gold \\ predicted |")


def test_perfect_classification():
    out = S.role_classification(["AI Engineer", "DevOps / Cloud"], ["AI Engineer", "DevOps / Cloud"])
    assert out["accuracy"] == 1.0 and out["macro_f1"] == 1.0


# --------------------------------------------------------------------------- retrieval
def test_requirement_retrieval_penalises_a_noun_dump():
    gold = ["python", "rag", "docker"]
    precise = S.requirement_retrieval(gold, ["Python", "RAG", "Docker"])
    assert precise == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    dump = S.requirement_retrieval(gold, ["Python", "RAG", "Docker"] + [f"noise{i}" for i in range(9)])
    assert dump["recall"] == 1.0
    assert dump["precision"] < 0.4, "recall alone cannot see a model that lists everything"
    assert dump["f1"] < precise["f1"]


def test_retrieval_matches_loosely_in_both_directions():
    out = S.requirement_retrieval(["gcp"], ["Google Cloud Platform (GCP)"])
    assert out["recall"] == 1.0


# --------------------------------------------------------------------------- frames
def arm(name: str, metrics: dict[str, float]) -> dict:
    return {
        "strategy": name,
        "metrics": {k: {"mean": v, "n": 1, "min": v} for k, v in metrics.items()},
        "usage": {"calls": 15, "prompt_tokens": 100, "completion_tokens": 50,
                  "total_tokens": 150, "cost_usd": 0.01, "p50_latency_s": 1.0},
        "wall_time_s": 10.0, "n_passed": 1, "n_thresholded": 2,
    }


def test_matrix_frame_is_metric_by_strategy():
    df = S.matrix_frame([arm("zero_shot", {"role_accuracy": 1.0}),
                         arm("few_shot", {"role_accuracy": 0.8})])
    assert list(df.columns) == ["zero_shot", "few_shot"]
    assert df.loc["role_accuracy", "few_shot"] == 0.8
    assert json.dumps(df.to_dict())  # must survive JSON serialisation into the report


def test_usage_frame_reports_per_strategy_economics():
    df = S.usage_frame([arm("zero_shot", {}), arm("few_shot", {})])
    assert df.loc["few_shot", "total_tokens"] == 150
    assert df.loc["zero_shot", "thresholds_met"] == "1/2"


# --------------------------------------------------------------------------- artifacts
def test_artifacts_round_trip(tmp_path, questions):
    log = ArtifactLog(tmp_path / "a.jsonl", run_id="test")
    log.write(strategy="few_shot", stage="questions", fixture="ai_engineer_junior",
              payload=questions, difficulty="hard")
    assert log.count == 1

    records = load_artifacts(tmp_path / "a.jsonl")
    assert len(records) == 1 and records[0]["strategy"] == "few_shot"

    flat = questions_from(records, strategy="few_shot")
    assert len(flat) == len(questions)
    assert flat[0]["difficulty"] == "hard"
    assert questions_from(records, strategy="zero_shot") == []


def test_artifacts_appended_incrementally(tmp_path, analysis):
    """A crash mid-run must keep what was already produced."""
    log = ArtifactLog(tmp_path / "a.jsonl", run_id="test")
    for i in range(3):
        log.write(strategy="zero_shot", stage="analysis", fixture=f"ad{i}", payload=analysis)
    assert len(load_artifacts(tmp_path / "a.jsonl")) == 3


def test_artifacts_redact_pii(tmp_path):
    log = ArtifactLog(tmp_path / "a.jsonl", run_id="test")
    log.write(strategy="zero_shot", stage="evaluation", fixture="x",
              payload={"answer": "reach me at jane.doe@example.com"})
    body = (tmp_path / "a.jsonl").read_text(encoding="utf-8")
    assert "jane.doe@example.com" not in body and "[email]" in body


def test_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"run":"a","stage":"analysis"}\n{truncated by a cra', encoding="utf-8")
    assert len(load_artifacts(path)) == 1


def test_disabled_log_writes_nothing(tmp_path):
    log = ArtifactLog(tmp_path / "a.jsonl", run_id="test", enabled=False)
    log.write(strategy="zero_shot", stage="analysis", fixture="x", payload={"a": 1})
    assert log.count == 0 and not (tmp_path / "a.jsonl").exists()


def test_unserialisable_payload_does_not_kill_the_run(tmp_path):
    log = ArtifactLog(tmp_path / "a.jsonl", run_id="test")
    log.write(strategy="zero_shot", stage="analysis", fixture="x", payload={"fn": object()})
    assert log.count == 1  # default=str keeps it writable rather than raising
