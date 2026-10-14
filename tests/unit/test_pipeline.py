"""Pipeline behaviour: grounding, concurrency, prompt wiring, failure tolerance."""
from __future__ import annotations

import time

import pytest

from interview_ai.pipeline import (
    NotAJobAd,
    analyze_job,
    assign_question_types,
    soft_skill_dimensions,
    ensure_soft_skills_topic,
    generate_questions,
    is_behavioural_topic,
    plan_interview,
    prepare_interview,
)
from interview_ai.prompts import PERSONAS, STRATEGIES, build_system
from interview_ai.domain import InterviewPlan, JobAnalysis, QuestionBatch, Topic
from tests.conftest import JOB_AD, FakeLLM


async def test_prepare_interview_runs_all_three_stages(fake_llm):
    analysis, plan, questions = await prepare_interview(fake_llm, JOB_AD, "medium", 8)
    assert analysis.role == "AI Engineer"
    assert sum(t.weight for t in plan.topics) == 100
    assert questions


async def test_non_job_ad_is_rejected(fake_llm, analysis):
    fake_llm.defaults[JobAnalysis] = JobAnalysis(is_job_ad=False)
    with pytest.raises(NotAJobAd):
        await analyze_job(fake_llm, "my grocery list: milk, eggs")


async def test_ad_with_no_requirements_is_rejected(fake_llm):
    fake_llm.defaults[JobAnalysis] = JobAnalysis(is_job_ad=True, title="Vibes Engineer")
    with pytest.raises(NotAJobAd):
        await analyze_job(fake_llm, "Come work with us! We have snacks.")


async def test_job_ad_is_sent_inside_untrusted_fence(fake_llm):
    await analyze_job(fake_llm, JOB_AD)
    prompt = fake_llm.prompts_for(JobAnalysis)[0]
    assert "<<<UNTRUSTED JOB_AD" in prompt and "UNTRUSTED JOB_AD>>>" in prompt
    assert "never instructions" in prompt


async def test_planner_prompt_carries_evidence_and_criticality(fake_llm, analysis):
    await plan_interview(fake_llm, analysis, "hard")
    planner = [c["user"] for c in fake_llm.calls if "Design the interview plan" in c["user"]][0]
    assert "Python :: critical" in planner
    assert "Python is non-negotiable" in planner
    assert "hard" in planner


async def test_planner_prompt_does_not_prescribe_topics(fake_llm, analysis):
    """The whole point of stage 2: nano derives the subjects, we never hand it a list."""
    await plan_interview(fake_llm, analysis, "medium")
    system = [c["system"] for c in fake_llm.calls][-1]
    for leak in ("Projects 25", "20% Python", "Always ask"):
        assert leak not in system


async def test_question_generation_is_concurrent(fake_llm, analysis, plan):
    started = time.perf_counter()
    await generate_questions(fake_llm, analysis, plan, "medium", 12)
    elapsed = time.perf_counter() - started
    # FakeLLM sleeps 10ms per call; 4 topics run together, so this stays well under serial time.
    assert fake_llm.max_parallel >= 2, "subject calls did not overlap"
    assert elapsed < 0.2


async def test_questions_are_interleaved_across_subjects(fake_llm, analysis, plan):
    questions = await generate_questions(fake_llm, analysis, plan, "medium", 8)
    first_topics = [q.topic for q in questions[:3]]
    assert len(set(first_topics)) == len(first_topics), "same subject asked back to back"


async def test_one_failing_subject_does_not_kill_the_interview(fake_llm, analysis, plan, questions):
    fake_llm.queue[QuestionBatch] = [
        RuntimeError("model hiccup"),
        QuestionBatch(questions=[questions[1]]),
        QuestionBatch(questions=[questions[2]]),
        QuestionBatch(questions=[questions[3]]),
    ]
    result = await generate_questions(fake_llm, analysis, plan, "medium", 8)
    assert result, "a single subject failure should not empty the bank"


async def test_topic_label_is_ours_not_the_models(fake_llm, analysis, plan):
    result = await generate_questions(fake_llm, analysis, plan, "medium", 8)
    assert {q.topic for q in result} <= {t.name for t in plan.topics}


async def test_strategy_reaches_every_stage_prompt(fake_llm, analysis, plan):
    """The strategy argument must survive all the way to the assembled system prompt."""
    await prepare_interview(fake_llm, JOB_AD, "medium", 8, "few_shot")
    systems = [c["system"] for c in fake_llm.calls]
    assert systems and all("--- EXAMPLE " in s for s in systems)

    fake_llm.calls.clear()
    await prepare_interview(fake_llm, JOB_AD, "medium", 8, "guided")
    systems = [c["system"] for c in fake_llm.calls]
    assert systems and all("PROCEDURE" in s and "--- EXAMPLE " not in s for s in systems)

    fake_llm.calls.clear()
    await prepare_interview(fake_llm, JOB_AD, "medium", 8, "self_critique")
    systems = [c["system"] for c in fake_llm.calls]
    assert systems and all(
        "SELF-CRITIQUE CHECKLIST" in s and "--- EXAMPLE " not in s for s in systems
    )


async def test_zero_shot_keeps_prompts_lean(fake_llm, analysis, plan):
    await prepare_interview(fake_llm, JOB_AD, "medium", 8, "zero_shot")
    zero = sum(len(c["system"]) for c in fake_llm.calls)
    fake_llm.calls.clear()
    await prepare_interview(fake_llm, JOB_AD, "medium", 8, "few_shot")
    few = sum(len(c["system"]) for c in fake_llm.calls)
    assert few > zero * 1.5, "few_shot should cost materially more prompt text than zero_shot"


async def test_per_stage_token_cap_is_sent(fake_llm, analysis, plan):
    await prepare_interview(fake_llm, JOB_AD, "medium", 8)
    stages = {c["kwargs"].get("stage") for c in fake_llm.calls}
    assert stages == {"analyzer", "planner", "questioner"}


async def test_difficulty_reaches_the_question_prompt(fake_llm, analysis, plan):
    await generate_questions(fake_llm, analysis, plan, "easy", 4)
    prompts = [c["user"] for c in fake_llm.calls if "SUBJECT:" in c["user"]]
    assert prompts and all("DIFFICULTY: easy" in p for p in prompts)
    assert any("Warm-up register" in p for p in prompts)


# --------------------------------------------------------------------------- type assignment
# Question-type distribution is a whole-interview constraint, and each subject's batch is a
# separate API call that cannot see the others -- so CODE assigns the types and the model
# only writes the questions. Measured before the fix: question_type_diversity 0.40-0.49 on
# every strategy arm (threshold 0.55), because every batch defaulted to the same 2-3 types.


def test_assigned_types_cover_the_taxonomy(plan):
    """One shared cycle across subjects must reach near-full type coverage by construction."""
    allocation = plan.question_allocation(8)
    assigned = assign_question_types(plan, allocation, "medium")
    flat = [t for types in assigned.values() for t in types]
    assert len(flat) == 8
    assert len(set(flat)) == 7, f"expected all 7 types across 8 slots, got {sorted(set(flat))}"


def test_behavioral_topic_gets_behavioral_types(plan):
    assigned = assign_question_types(plan, plan.question_allocation(8), "medium")
    assert assigned["Collaboration"][0] == "behavioral"
    # Technical subjects never receive the behavioural type -- it belongs to its subject.
    technical = [t for name, types in assigned.items() if name != "Collaboration" for t in types]
    assert "behavioral" not in technical


def test_difficulty_tilts_the_type_cycle(plan):
    allocation = plan.question_allocation(8)
    easy = assign_question_types(plan, allocation, "easy")
    hard = assign_question_types(plan, allocation, "hard")
    first_topic = plan.topics[0].name
    assert easy[first_topic][0] == "conceptual"
    assert hard[first_topic][0] == "scenario"


def test_assignment_skips_topics_with_no_questions(plan):
    allocation = {plan.topics[0].name: 2}  # only one subject got questions
    assigned = assign_question_types(plan, allocation, "medium")
    assert set(assigned) == {plan.topics[0].name}


async def test_assigned_types_reach_the_prompt_and_the_output(fake_llm, analysis, plan):
    """The prompt carries TYPES TO WRITE, and the returned questions carry the assignment
    even when the model labelled them differently -- the slot's type is our bookkeeping."""
    result = await generate_questions(fake_llm, analysis, plan, "medium", 8)

    prompts = [c["user"] for c in fake_llm.calls if "SUBJECT:" in c["user"]]
    assert prompts and all("TYPES TO WRITE" in p for p in prompts)

    expected = assign_question_types(plan, plan.question_allocation(8), "medium")
    for q in result:
        assert q.type == expected[q.topic][0], (
            f"{q.topic}: model label survived instead of the assigned type"
        )


# --------------------------------------------------------------------------- soft skills
# Reported failure: every "soft skills" question was really a technical one, because the
# planner satisfied its behavioural-subject rule with a subject like "Team-based CI/CD".


def test_technical_subject_wearing_a_people_name_is_not_behavioural():
    assert is_behavioural_topic("Collaboration & communication") is True
    assert is_behavioural_topic("Stakeholder communication") is True
    assert is_behavioural_topic("Team collaboration on CI/CD pipelines") is False
    assert is_behavioural_topic("Cross-team API design") is False


def test_a_real_soft_skills_subject_is_injected_when_the_plan_lacks_one():
    plan = InterviewPlan(topics=[
        Topic(name="RAG & retrieval", weight=40, rationale="x", grounded_in=["RAG"]),
        Topic(name="Python services", weight=30, rationale="x", grounded_in=["Python"]),
        Topic(name="Team collaboration on CI/CD pipelines", weight=30, rationale="x",
              grounded_in=["CI"]),
    ])
    fixed = ensure_soft_skills_topic(plan)
    assert any(is_behavioural_topic(t.name) for t in fixed.topics)
    assert sum(t.weight for t in fixed.topics) == 100, "weights must still validate"


def test_an_existing_real_soft_skills_subject_is_left_alone(plan):
    assert ensure_soft_skills_topic(plan).topics == plan.topics


async def test_soft_skills_batch_is_told_not_to_ask_about_technology(fake_llm, analysis, plan):
    await generate_questions(fake_llm, analysis, plan, "medium", 8)
    soft = [c["user"] for c in fake_llm.calls
            if "SUBJECT: Collaboration" in c["user"]]
    assert soft, "the behavioural subject got no question batch"
    assert "SOFT SKILLS" in soft[0] and "Do NOT ask about any technology" in soft[0]
    technical = [c["user"] for c in fake_llm.calls if "SUBJECT: LLM & RAG" in c["user"]]
    assert technical and "TECHNICAL --" in technical[0]


# --------------------------------------------------------------------------- personas
def test_persona_changes_tone_only_where_the_candidate_hears_it():
    for stage in ("analyzer", "planner", "reporter"):
        assert build_system(stage, "zero_shot", persona="strict") == build_system(stage, "zero_shot")
    for persona in ("friendly", "strict"):
        q = build_system("questioner", "zero_shot", persona=persona, n=2)
        assert "INTERVIEWER MANNER" in q
        # The rules must survive underneath: a persona appends, never replaces.
        assert q.startswith(build_system("questioner", "zero_shot", n=2))


def test_unknown_persona_is_rejected():
    with pytest.raises(ValueError, match="Unknown persona"):
        build_system("questioner", "zero_shot", persona="sarcastic", n=2)


async def test_persona_reaches_the_question_prompt(fake_llm, analysis, plan):
    await generate_questions(fake_llm, analysis, plan, "medium", 4, persona="strict")
    systems = [c["system"] for c in fake_llm.calls if "You are the interviewer" in c["system"]]
    assert systems and all("INTERVIEWER MANNER: strict" in s for s in systems)


# --------------------------------------------------------------------------- verbosity
def test_standard_verbosity_leaves_every_prompt_byte_identical():
    """The default must not move the strategy comparison: no block, no change."""
    for stage in ("analyzer", "planner", "evaluator", "reporter"):
        fmt = {"max_followups": 1} if stage == "evaluator" else {}
        for strategy in STRATEGIES:
            assert (build_system(stage, strategy, verbosity="standard", **fmt)
                    == build_system(stage, strategy, **fmt))


def test_verbosity_changes_length_only_where_the_candidate_reads_it():
    for stage in ("analyzer", "planner"):
        assert build_system(stage, "zero_shot", verbosity="concise") == build_system(stage, "zero_shot")
    # The scorer is deliberately excluded: a length preference must not move a score.
    assert (build_system("evaluator", "zero_shot", verbosity="detailed", max_followups=1)
            == build_system("evaluator", "zero_shot", max_followups=1))
    for verbosity in ("concise", "detailed"):
        for stage, fmt in (("questioner", {"n": 2}), ("reporter", {})):
            text = build_system(stage, "zero_shot", verbosity=verbosity, **fmt)
            assert f"RESPONSE LENGTH: {verbosity}" in text
            assert text.startswith(build_system(stage, "zero_shot", **fmt))


def test_unknown_verbosity_is_rejected():
    with pytest.raises(ValueError, match="Unknown verbosity"):
        build_system("reporter", "zero_shot", verbosity="epic")


async def test_verbosity_reaches_the_question_prompt(fake_llm, analysis, plan):
    await generate_questions(fake_llm, analysis, plan, "medium", 4, verbosity="concise")
    systems = [c["system"] for c in fake_llm.calls if "You are the interviewer" in c["system"]]
    assert systems and all("RESPONSE LENGTH: concise" in s for s in systems)


# --------------------------------------------------------------------------- personality
# Soft skills are measured as personality and working style, so the dimensions are assigned
# by code (two behavioural questions must not both be "tell me about a conflict") and the
# questions must not imply a correct temperament.


def test_soft_skill_dimensions_do_not_repeat():
    dims = soft_skill_dimensions(4)
    assert len({name for name, _ in dims}) == 4


def test_soft_skills_batch_names_the_personality_dimensions(plan):
    from interview_ai.pipeline.generate_questions import _subject_kind

    behavioural = next(t for t in plan.topics if is_behavioural_topic(t.name))
    kind = _subject_kind(behavioural, 2)
    assert "PERSONALITY" in kind
    assert "Team vs solo working style" in kind
    assert "NO right answer" in kind
    assert "Do NOT ask about any technology" in kind


def test_behavioural_slots_all_take_the_behavioural_type(plan):
    """Variety across soft-skills questions comes from the dimension, not the question form."""
    allocation = dict.fromkeys([t.name for t in plan.topics], 2)
    assigned = assign_question_types(plan, allocation, "medium")
    behavioural = next(t.name for t in plan.topics if is_behavioural_topic(t.name))
    assert set(assigned[behavioural]) == {"behavioral"}


async def test_evaluator_is_told_a_soft_skills_question_has_no_right_answer(
    fake_llm, analysis, plan, questions
):
    from interview_ai.domain import InterviewSession

    behavioural = [q for q in questions if q.type == "behavioral"]
    assert behavioural, "fixture should include a behavioural question"
    s = InterviewSession(analysis=analysis, plan=plan, questions=behavioural, difficulty="medium")
    await s.submit_answer(fake_llm, "I prefer working alone, but I pair when the design is unclear.")
    prompt = fake_llm.calls[-1]["user"] + fake_llm.calls[-1]["system"]
    assert "QUESTION KIND: SOFT SKILLS" in prompt
    assert "there is no correct answer" in prompt.lower()
    assert "Never mark an answer down for the personality it describes." in prompt
