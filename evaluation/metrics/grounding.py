"""Is the output tied to the input, or invented?

The anti-hallucination half of the suite. Every metric here asks whether what the model
produced can be traced back to the job ad it was given -- and the exemplar-leakage metrics
at the bottom ask the mirror-image question: did it come from a worked example instead?
"""
from __future__ import annotations

from interview_ai.domain import InterviewPlan, JobAnalysis, Question
from interview_ai.prompts import EXEMPLAR_FINGERPRINTS

from ._shared import SENIORITY_ORDER, mentions, norm


def role_correct(analysis: JobAnalysis, gold_role: str) -> float:
    return float(analysis.role.strip().lower() == gold_role.strip().lower())


def seniority_exact(analysis: JobAnalysis, gold: str) -> float:
    return float(analysis.seniority == gold)


def seniority_within_one(analysis: JobAnalysis, gold: str) -> float:
    try:
        return float(abs(SENIORITY_ORDER.index(analysis.seniority) - SENIORITY_ORDER.index(gold)) <= 1)
    except ValueError:
        return 0.0


def requirement_recall(analysis: JobAnalysis, must_have: list[str]) -> float:
    """Fraction of gold must-have skills that appear anywhere in the extraction."""
    if not must_have:
        return 1.0
    blob = " | ".join([r.name for r in analysis.requirements] + analysis.technologies)
    return sum(mentions(m, blob) for m in must_have) / len(must_have)


def evidence_grounding(analysis: JobAnalysis, ad_text: str) -> float:
    """Fraction of requirements whose `evidence` quote is genuinely present in the ad.

    This is the anti-hallucination metric: a fabricated requirement cannot be quoted.
    """
    if not analysis.requirements:
        return 0.0
    ad = norm(ad_text)
    hits = 0
    for r in analysis.requirements:
        quote = norm(r.evidence)
        if not quote:
            continue
        if quote in ad:
            hits += 1
        else:  # tolerate light paraphrase: most of the quoted words must be in the ad
            words = [w for w in quote.split() if len(w) > 2]
            if words and sum(w in ad for w in words) / len(words) >= 0.8:
                hits += 1
    return hits / len(analysis.requirements)


def criticality_precision(analysis: JobAnalysis, gold_critical: list[str]) -> float:
    """Of the gold hard-gate skills, how many were actually marked `critical`?"""
    if not gold_critical:
        return 1.0
    critical = " | ".join(r.name for r in analysis.requirements if r.criticality == "critical")
    return sum(mentions(g, critical) for g in gold_critical) / len(gold_critical)


def emphasis_detected(analysis: JobAnalysis) -> float:
    return float(bool(analysis.emphasis))


# --------------------------------------------------------------------------- stage 2


def topic_grounding(plan: InterviewPlan, analysis: JobAnalysis) -> float:
    """Fraction of subjects that name at least one real requirement in `grounded_in`."""
    if not plan.topics:
        return 0.0
    known = " | ".join([r.name for r in analysis.requirements] + analysis.technologies)
    grounded = sum(
        bool(t.grounded_in) and any(mentions(g, known) for g in t.grounded_in)
        for t in plan.topics
    )
    return grounded / len(plan.topics)


def syllabus_coverage(plan: InterviewPlan, expected_keywords: list[str]) -> float:
    """Does the derived syllabus actually cover what the job is about?"""
    if not expected_keywords:
        return 1.0
    blob = " ".join(f"{t.name} {t.rationale}" for t in plan.topics)
    return sum(mentions(k, blob) for k in expected_keywords) / len(expected_keywords)


def question_grounding(questions: list[Question], analysis: JobAnalysis) -> float:
    """Fraction of questions that target a requirement the ad actually stated."""
    if not questions:
        return 0.0
    known = " | ".join([r.name for r in analysis.requirements] + analysis.technologies)
    return sum(
        bool(q.targets) and any(mentions(t, known) for t in q.targets) for q in questions
    ) / len(questions)


def rubric_present(questions: list[Question]) -> float:
    if not questions:
        return 0.0
    return sum(len(q.what_good_looks_like.strip()) > 20 for q in questions) / len(questions)


# The cost side of few-shot prompting: a small model may reproduce the worked example instead
# of reading the input. Exemplars use roles absent from the fixtures, so any appearance of
# exemplar-specific vocabulary in output for a fixture ad is copying, not coincidence.
def _leakable_terms(ad_text: str) -> list[str]:
    """Exemplar vocabulary minus anything the ad itself says.

    Without this filter the metric is unsound. The `frontend_mid` fixture genuinely lists
    "Playwright" in its testing requirements, so echoing it is correct behaviour, not copying
    -- yet the unfiltered check flagged it on every arm including zero_shot, which has no
    exemplars in its prompt at all.
    """
    if not ad_text:
        return list(EXEMPLAR_FINGERPRINTS["terms"])
    return [t for t in EXEMPLAR_FINGERPRINTS["terms"] if not mentions(t, ad_text)]


def analysis_exemplar_leakage(analysis: JobAnalysis, gold_role: str, ad_text: str = "") -> float:
    """1.0 = clean, lower = the analysis carries content copied from a worked example."""
    hits = 0.0
    blob = " | ".join(
        [analysis.title, analysis.domain]
        + [r.name for r in analysis.requirements]
        + analysis.technologies
        + analysis.responsibilities
        + analysis.emphasis
    )
    # Wrong role, and specifically an exemplar's role, is the strongest signal.
    if analysis.role != gold_role and analysis.role in EXEMPLAR_FINGERPRINTS["roles"]:
        hits += 1
    hits += sum(mentions(term, blob) for term in _leakable_terms(ad_text))
    return 1.0 if hits == 0 else max(0.0, 1.0 - hits / 3)


def plan_exemplar_leakage(plan: InterviewPlan, ad_text: str = "") -> float:
    """1.0 = clean. Detects copied subject names and exemplar vocabulary."""
    names = {norm(t.name) for t in plan.topics}
    copied = sum(norm(t) in names for t in EXEMPLAR_FINGERPRINTS["topics"])

    blob = " ".join(f"{t.name} {t.rationale}" for t in plan.topics)
    term_hits = sum(mentions(term, blob) for term in _leakable_terms(ad_text))

    # Every planner exemplar uses the 30/25/20/15/10 multiset, so reproducing it *looks* like
    # copying -- but the first full matrix fired this on zero_shot, whose prompt contains no
    # examples to copy. A smooth descending split is simply what any sensible weighting looks
    # like. So the shape now only corroborates: it counts when there is already a name or
    # vocabulary hit, and never accuses on its own.
    exemplar_shape = sorted([30, 25, 20, 15, 10], reverse=True)
    shape_copied = sorted((t.weight for t in plan.topics), reverse=True) == exemplar_shape

    hits = copied + term_hits
    if shape_copied and hits:
        hits += 1
    return 1.0 if hits == 0 else max(0.0, 1.0 - hits / 3)


def question_exemplar_leakage(questions: list[Question], ad_text: str = "") -> float:
    """1.0 = clean. Detects exemplar technologies leaking into questions for a different job."""
    if not questions:
        return 1.0
    blob = " ".join(q.text for q in questions)
    hits = sum(mentions(term, blob) for term in _leakable_terms(ad_text))
    return 1.0 if hits == 0 else max(0.0, 1.0 - hits / 3)
