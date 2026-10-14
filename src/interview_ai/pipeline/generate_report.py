"""Stage 5: the final report."""
from __future__ import annotations

from ..config import SETTINGS
from ..domain.interview import InterviewSession
from ..domain.scoring import FinalReport
from ..llm import LLM
from ..prompts import Strategy, build_system, render, user_template
from ..utils.security import redact_pii


async def build_report(
    llm: LLM, session: InterviewSession, strategy: Strategy | None = None
) -> FinalReport:
    a = session.analysis
    requirements = "\n".join(
        f"- {r.name} ({r.criticality})"
        for r in a.requirements
        if r.criticality in ("critical", "important")
    ) or "- (none extracted)"
    plan = "\n".join(f"- {t.name} :: {t.weight}%" for t in session.plan.topics)

    return await llm.structured(
        FinalReport,
        build_system(
            "reporter",
            strategy or session.strategy or SETTINGS.strategy_for("reporter"),
            verbosity=session.verbosity or session.settings.verbosity,
        ),
        render(
            user_template("reporter"),
            title=a.title or "(untitled)",
            company=a.company or "(unnamed company)",
            role=a.role,
            seniority=a.seniority,
            difficulty=session.difficulty,
            requirements=requirements,
            plan=plan,
            transcript=session.transcript_for_model(),
            # Coverage is measured by the state machine against the difficulty threshold,
            # so `unmet_requirements` is evidence the report must respect rather than a
            # judgement the model re-derives from the transcript.
            coverage=session.coverage_summary(),
            unmet_measured=", ".join(session.unmet_requirements()) or "(none -- every question reached its threshold)",
        ),
        stage="reporter",
    )


RECOMMENDATION_LABEL = {
    "strong_yes": "Strong yes -- move to the next round",
    "yes": "Yes -- move to the next round",
    "borderline": "Borderline -- would need another signal",
    "no": "No -- not for this role yet",
}


def report_markdown(report: FinalReport, session: InterviewSession) -> str:
    """Downloadable report. PII-redacted, since it contains the candidate transcript."""
    a = session.analysis
    parts = [
        f"# Interview report -- {a.title or a.role}",
        f"*{a.role} - {a.seniority} - {session.difficulty} difficulty*",
        "",
        f"**Overall score:** {report.overall_score:.0f}/100",
        f"**Recommendation:** {RECOMMENDATION_LABEL.get(report.recommendation, report.recommendation)}",
        "",
        report.recommendation_reason,
        "",
        "## Score by subject",
        *[f"- **{t.topic}** - {t.score:.1f}/5 - {t.comment}" for t in report.per_topic],
        "",
        "## Strengths",
        *([f"- {s}" for s in report.strengths] or ["- (none recorded)"]),
        "",
        "## Weaknesses",
        *([f"- {w}" for w in report.weaknesses] or ["- (none recorded)"]),
        "",
        "## Job requirements you did not demonstrate",
        *([f"- {r}" for r in report.unmet_requirements] or ["- (all covered)"]),
        "",
        "## Questions you struggled with",
        *([f"- {q}" for q in report.struggled_questions] or ["- (none)"]),
        "",
        "## Study before the real interview",
        *([f"{i}. {s}" for i, s in enumerate(report.study_plan, 1)] or ["1. (nothing flagged)"]),
        "",
        "---",
        "",
        session.transcript_for_export(),
    ]
    return redact_pii("\n".join(parts))
