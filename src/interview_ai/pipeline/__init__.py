"""Job ad -> analysis -> plan -> questions -> scored answers -> report.

One module per stage, in the order the interview runs them:

    analyze_job         stage 1  read the ad, extract requirements with evidence
    build_plan          stage 2  weight the syllabus, guarantee a soft-skills subject
    generate_questions  stage 3  allocate slots, assign types, fan out one call per subject
    score_answer        stage 4  grade one answer and decide whether to follow up
    generate_report     stage 5  the final debrief

Every stage takes a `strategy` (zero_shot / one_shot / few_shot / guided / self_critique)
and defaults to
whatever `Settings.strategy_for(stage)` says, so the app respects `.env` while the eval
harness can force one strategy across a whole sweep.
"""
from __future__ import annotations

from ..config import SETTINGS, Difficulty
from ..domain.interview import InterviewPlan
from ..domain.job import JobAnalysis
from ..domain.question import Question
from ..llm import LLM
from ..prompts import Strategy
from .analyze_job import NotAJobAd, analyze_job
from .build_plan import (
    SOFT_SKILLS_RATIONALE,
    SOFT_SKILLS_TOPIC,
    ensure_soft_skills_topic,
    is_behavioural_topic,
    plan_interview,
)
from .generate_questions import (
    SOFT_SKILL_DIMENSIONS,
    assign_question_types,
    ensure_behavioural_slot,
    generate_questions,
    soft_skill_dimensions,
)
from .generate_report import RECOMMENDATION_LABEL, build_report, report_markdown
from .score_answer import score_answer

__all__ = [
    "NotAJobAd", "RECOMMENDATION_LABEL", "SOFT_SKILLS_RATIONALE", "SOFT_SKILLS_TOPIC",
    "SOFT_SKILL_DIMENSIONS", "analyze_job", "assign_question_types", "build_report",
    "ensure_behavioural_slot", "ensure_soft_skills_topic", "generate_questions",
    "is_behavioural_topic", "plan_interview", "prepare_interview", "report_markdown",
    "score_answer", "soft_skill_dimensions",
]


async def prepare_interview(
    llm: LLM,
    job_ad: str,
    difficulty: Difficulty,
    total_questions: int = SETTINGS.default_questions,
    strategy: Strategy | None = None,
    persona: str | None = None,
    verbosity: str | None = None,
) -> tuple[JobAnalysis, InterviewPlan, list[Question]]:
    """Full preparation: analyse -> plan -> questions."""
    analysis = await analyze_job(llm, job_ad, strategy)
    plan = ensure_soft_skills_topic(await plan_interview(llm, analysis, difficulty, strategy))
    questions = await generate_questions(
        llm, analysis, plan, difficulty, total_questions, strategy, persona, verbosity
    )
    return analysis, plan, questions
