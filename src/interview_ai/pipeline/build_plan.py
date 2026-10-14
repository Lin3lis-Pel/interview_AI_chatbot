"""Stage 2: turn the analysis into an interview syllabus.

The planner is REQUIRED to produce exactly one behavioural subject. It complies on paper
and repeatedly fails in substance, so the guarantees the interview depends on are enforced
here in code rather than asked of the model.
"""
from __future__ import annotations

import re

from ..config import DIFFICULTY_RUBRIC, SENIORITY_RUBRIC, SETTINGS, Difficulty
from ..domain.interview import InterviewPlan, Topic
from ..domain.job import JobAnalysis
from ..llm import LLM
from ..prompts import Strategy, build_system, render, user_template
from ..utils.text import bullets, requirement_lines


async def plan_interview(
    llm: LLM, analysis: JobAnalysis, difficulty: Difficulty, strategy: Strategy | None = None
) -> InterviewPlan:
    return await llm.structured(
        InterviewPlan,
        build_system("planner", strategy or SETTINGS.strategy_for("planner")),
        render(
            user_template("planner"),
            role=analysis.role,
            seniority=analysis.seniority,
            seniority_rubric=SENIORITY_RUBRIC.get(analysis.seniority, ""),
            difficulty=difficulty,
            difficulty_rubric=DIFFICULTY_RUBRIC[difficulty],
            domain=analysis.domain or "(not stated)",
            emphasis=", ".join(analysis.emphasis) or "(nothing stands out)",
            requirements=requirement_lines(analysis),
            responsibilities=bullets(analysis.responsibilities),
        ),
        stage="planner",
    )


# A genuine soft-skills subject, not a technical one that happens to contain "team".
# Measured problem: the planner names subjects like "Team-based CI/CD ownership", the old
# regex matched the word "team", and the interview then counted a pipeline question as its
# soft-skills coverage. So the match now requires a people word, and any subject naming a
# technology or engineering practice is rejected outright however it is worded.
_BEHAVIORAL_TOPIC = re.compile(
    r"behaviou?r|soft skill|interpersonal|communicat|collaborat(?:ion|ing)|teamwork|"
    r"stakeholder|conflict|feedback|mentor|leadership|culture|ways of working|working with people|people",
    re.IGNORECASE,
)
_TECHNICAL_MARKER = re.compile(
    r"pipeline|ci/?cd|deploy|architect|code|coding|test|debug|data|model|api|cloud|"
    r"infrastructure|security|python|sql|docker|kubernetes|rag|llm|design|system|"
    r"performance|monitor|database|retrieval|engineering",
    re.IGNORECASE,
)

SOFT_SKILLS_TOPIC = "Working with people"
SOFT_SKILLS_RATIONALE = (
    "Every role here is delivered with other people: this subject probes communication, "
    "collaboration, disagreement and ownership, never technology."
)

def is_behavioural_topic(name: str) -> bool:
    """True only for a subject about working with people, not about technology."""
    return bool(_BEHAVIORAL_TOPIC.search(name)) and not _TECHNICAL_MARKER.search(name)


def ensure_soft_skills_topic(plan: InterviewPlan) -> InterviewPlan:
    """Guarantee the plan contains a genuine soft-skills subject, injecting one if not.

    The planner is told to produce exactly one behavioural subject, and it complies on
    paper -- but it repeatedly satisfies the rule with a technical subject wearing a
    people-ish name ("Team-based CI/CD ownership"), which then receives pipeline questions.
    Detection alone cannot fix that: if no real subject exists, one has to be added.

    Weight is taken from the largest subject so the total stays exactly 100, which the
    schema validates.
    """
    if any(is_behavioural_topic(t.name) for t in plan.topics):
        return plan
    topics = [t.model_copy() for t in plan.topics]
    largest = max(topics, key=lambda t: t.weight)
    weight = min(10, max(5, largest.weight - 5))
    if largest.weight - weight < 5:
        return plan                     # cannot fund it without gutting a real subject
    largest.weight -= weight
    topics.append(Topic(
        name=SOFT_SKILLS_TOPIC, weight=weight, rationale=SOFT_SKILLS_RATIONALE,
        grounded_in=list(largest.grounded_in),
    ))
    return InterviewPlan(topics=topics)
