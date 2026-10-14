"""Stage 3: write the question bank.

Generation fans out: one concurrent call per subject, so a 6-subject interview is prepared
in roughly the time of a single call. What no single call can see -- the shape of the whole
interview -- is decided here in code before any call is made: how many questions each
subject gets, which type each slot takes, and which personality dimension each soft-skills
question probes.
"""
from __future__ import annotations

import asyncio
import itertools

from ..config import DIFFICULTY_RUBRIC, SENIORITY_RUBRIC, SETTINGS, Difficulty
from ..domain.interview import InterviewPlan, Topic
from ..domain.job import JobAnalysis
from ..domain.question import Question, QuestionBatch
from ..llm import LLM
from ..llm.structured_output import LLMUnavailable
from ..prompts import Strategy, build_system, render, user_template
from ..utils.text import bullets
from .build_plan import ensure_soft_skills_topic, is_behavioural_topic


# The personality dimensions a soft-skills subject actually probes. Distribution is owned
# by code for the same reason question types are: each subject's batch is one API call, so
# left to itself the model asks "tell me about a conflict" twice and calls it variety.
#
# None of these has a right answer. Someone who prefers solo deep work is not worse than
# someone who thrives in a group -- what is being measured is whether they know how they
# work, can name the trade-off, and have a real situation to point at.
SOFT_SKILL_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("Team vs solo working style",
     "how they prefer to work, what they get from a group and what they need alone, and "
     "which kind of work suits each"),
    ("Handling disagreement",
     "how they argue a position they believe in, and what makes them change their mind or "
     "concede and commit"),
    ("Friction with a colleague",
     "a real disagreement that got personal or stuck, and how it actually resolved"),
    ("Giving and taking feedback",
     "how they deliver criticism to a peer, and how they respond to receiving it"),
    ("Owning a mistake",
     "something they got wrong, who they told, when, and what changed afterwards"),
    ("Pressure and shifting priorities",
     "how they behave when the deadline moves or the plan changes under them"),
    ("Explaining to non-technical people",
     "how they get a technical constraint across to someone who does not share the vocabulary"),
    ("Autonomy and asking for help",
     "how long they push alone before asking, and how they decide"),
)


def soft_skill_dimensions(n: int, offset: int = 0) -> list[tuple[str, str]]:
    """`n` distinct personality dimensions, so two soft-skills questions never overlap."""
    dims = SOFT_SKILL_DIMENSIONS
    return [dims[(offset + i) % len(dims)] for i in range(n)]

# Technical types in the order each difficulty reaches for first. The assignment walks ONE
# cycle across ALL subjects -- which is exactly what generation cannot do on its own: each
# subject's batch is a separate API call that sees no other batch, so left to the model
# every batch defaults to the same 2-3 types (measured: question_type_diversity 0.40-0.49
# on all four strategy arms in the 2026-08-21 run). Distribution is a constraint, so code owns
# it; the model only writes the question.
_TYPE_CYCLE: dict[str, tuple[str, ...]] = {
    "easy": ("conceptual", "practical", "project_experience", "scenario", "debugging", "architecture"),
    "medium": ("practical", "scenario", "conceptual", "debugging", "project_experience", "architecture"),
    "hard": ("scenario", "debugging", "architecture", "practical", "project_experience", "conceptual"),
}


def ensure_behavioural_slot(plan: InterviewPlan, allocation: dict[str, int]) -> dict[str, int]:
    """Guarantee the behavioural subject at least one question on any real interview.

    The planner is REQUIRED to produce exactly one behavioural subject, weighted 5-15%.
    Largest-remainder apportionment then throws it away on short interviews: 10% of 5
    questions is 0.5, which loses every tiebreak to a bigger subject. Measured across
    realistic weightings, soft skills got ZERO questions at n=5 every time -- the interview
    silently dropped the only subject the planner was told it must include.

    So the floor is enforced here, taking the question from the largest subject (which can
    spare one) rather than letting the rounding decide.
    """
    total = sum(allocation.values())
    behavioural = [t.name for t in plan.topics if is_behavioural_topic(t.name)]
    if not behavioural or total < 4:
        return allocation           # too short to spend a slot on anything but the job
    name = behavioural[0]
    if allocation.get(name, 0) > 0:
        return allocation

    donors = sorted(
        (n for n in allocation if n != name and allocation[n] > 1),
        key=lambda n: allocation[n], reverse=True,
    )
    if not donors:
        return allocation           # nobody can spare one without emptying a subject
    allocation = dict(allocation)
    allocation[donors[0]] -= 1
    allocation[name] = 1
    return allocation


def assign_question_types(
    plan: InterviewPlan, allocation: dict[str, int], difficulty: Difficulty
) -> dict[str, list[str]]:
    """Deterministically assign a question type to every slot of the interview.

    The behavioural subject alternates behavioural/experience questions; technical
    subjects draw from one shared cycle spanning the whole interview, tilted by
    difficulty (easy leads with definitions, hard with changed-constraint scenarios).
    """
    cycle = itertools.cycle(_TYPE_CYCLE[difficulty])
    assigned: dict[str, list[str]] = {}
    for topic in plan.topics:
        n = allocation.get(topic.name, 0)
        if n <= 0:
            continue
        if is_behavioural_topic(topic.name):
            # Every soft-skills question is type `behavioral`; what varies between them is
            # the personality dimension, not the question form.
            assigned[topic.name] = ["behavioral"] * n
        else:
            assigned[topic.name] = [next(cycle) for _ in range(n)]
    return assigned


def _subject_kind(topic: Topic, n: int) -> str:
    """What kind of subject this batch is, and for soft skills, exactly what to probe."""
    if not is_behavioural_topic(topic.name):
        return "TECHNICAL -- ask about the work itself, grounded in the requirements above."

    dims = soft_skill_dimensions(n)
    listed = "\n".join(f"    {i}. {name} -- {probe}" for i, (name, probe) in enumerate(dims, 1))
    return (
        "SOFT SKILLS -- this subject measures PERSONALITY and working style, not knowledge.\n"
        "Probe these dimensions, one per question, in this order:\n"
        f"{listed}\n"
        "    Ask about the person: how they work with others, what they do when people "
        "disagree, how they handle being wrong. Do NOT ask about any technology, tool, "
        "architecture or process here -- a technical question in this subject leaves the "
        "interview with no personality signal at all.\n"
        "    There is NO right answer to these. Do not write a question that implies one "
        "(never 'how do you make sure the team always agrees?'). Ask what they actually do, "
        "and `what_good_looks_like` must describe a SUBSTANTIVE answer -- a specific "
        "situation, what they personally did, and awareness of the trade-off -- never a "
        "preferred personality."
    )


async def _questions_for_topic(
    llm: LLM,
    analysis: JobAnalysis,
    topic: Topic,
    difficulty: Difficulty,
    qtypes: list[str],
    already_asked: list[str],
    strategy: str,
    persona: str = "neutral",
    verbosity: str = "standard",
) -> list[Question]:
    batch = await llm.structured(
        QuestionBatch,
        build_system("questioner", strategy, persona=persona, verbosity=verbosity,
                     n=len(qtypes)),
        render(
            user_template("questioner"),
            topic=topic.name,
            rationale=topic.rationale,
            grounded_in=", ".join(topic.grounded_in) or "(see requirements)",
            qtypes=", ".join(qtypes),
            subject_kind=_subject_kind(topic, len(qtypes)),
            role=analysis.role,
            seniority=analysis.seniority,
            seniority_rubric=SENIORITY_RUBRIC.get(analysis.seniority, ""),
            difficulty=difficulty,
            difficulty_rubric=DIFFICULTY_RUBRIC[difficulty],
            technologies=", ".join(analysis.technologies) or "(none named)",
            already_asked=bullets(already_asked, "(nothing yet)"),
        ),
        stage="questioner",
    )
    questions = batch.questions[: len(qtypes)]
    for q, assigned_type in zip(questions, qtypes):
        q.topic = topic.name       # trust our own bookkeeping over the model's
        q.type = assigned_type     # the type is the slot's assignment, not the model's pick
    return questions


async def generate_questions(
    llm: LLM,
    analysis: JobAnalysis,
    plan: InterviewPlan,
    difficulty: Difficulty,
    total: int = SETTINGS.default_questions,
    strategy: Strategy | None = None,
    persona: str | None = None,
    verbosity: str | None = None,
) -> list[Question]:
    """Fan out one call per subject, then interleave so subjects alternate in the interview."""
    plan = ensure_soft_skills_topic(plan)
    allocation = ensure_behavioural_slot(plan, plan.question_allocation(total))
    # Types are assigned HERE, across the whole interview, before any call is made --
    # a per-subject call cannot balance what it cannot see.
    type_plan = assign_question_types(plan, allocation, difficulty)
    topics = {t.name: t for t in plan.topics}
    strat = strategy or SETTINGS.strategy_for("questioner")
    manner = persona or SETTINGS.persona
    length = verbosity or SETTINGS.verbosity
    names = [name for name in allocation if name in topics]

    results = await asyncio.gather(
        *(
            _questions_for_topic(llm, analysis, topics[name], difficulty,
                                 type_plan[name], [], strat, manner, length)
            for name in names
        ),
        return_exceptions=True,
    )

    buckets: list[list[Question]] = []
    failures: list[str] = []
    for name, res in zip(names, results):
        if isinstance(res, Exception):
            failures.append(f"{name}: {res}")
        elif res:
            buckets.append(list(res))

    if not buckets:
        raise LLMUnavailable("No interview questions could be generated. " + " | ".join(failures))

    # Round-robin so the candidate is not asked six Python questions in a row.
    ordered: list[Question] = []
    while any(buckets):
        for bucket in buckets:
            if bucket:
                ordered.append(bucket.pop(0))
    return ordered
