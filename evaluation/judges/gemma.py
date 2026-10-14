"""Independent second judge, for the eval harness only.

Why a second model at all
-------------------------
In the product, the same `gpt-5-nano` writes the question, writes the rubric, and then scores
the answer against its own rubric. That is self-preference bias, and no code metric can see
it: if the model is systematically generous to answers phrased the way it phrases things,
every score moves together and the ordering still looks fine.

An independent judge on a *different model family* breaks that. Where the two disagree is
where the primary score is doing something other than measuring the answer.

Setup (local Ollama, no cost per run)
-------------------------------------
    ollama list          # take the exact tag from here
    ollama serve

    # config/settings.toml [judge] (commented out by default) -- `model` must match the tag
    # Ollama reports, verbatim
    [judge]
    model = "gemma4:e4b"
    base_url = "http://localhost:11434/v1"

Ollama speaks the OpenAI wire format, so this reuses `interview_ai.llm.LLM` unchanged -- only
`model`, `base_url` and `api_key` are overridden. The SSRF guard in interview_ai/utils/security.py applies
to job-ad URL fetching, not to this endpoint, so localhost is fine here.

Judge quality caveat
--------------------
A small local model is a weak judge of deep technical correctness. Treat it as a *second
opinion*, not an authority: the useful signal is agreement and disagreement with the primary,
plus the question-quality checks (grounded / realistic / single-ask), which are judgement
calls a small model handles fine and regex cannot handle at all. The gold-tier score bands in
`evaluation/fixtures/answers/` remain the ground truth.
"""
from __future__ import annotations

import asyncio

from interview_ai.config import SETTINGS
from interview_ai.llm import LLM, LLMUnavailable
from interview_ai.prompts import build_system, render, user_template
from interview_ai.domain import JobAnalysis, JudgeAnswerScore, JudgeQuestionRating, Question


class JudgeUnavailable(RuntimeError):
    """No judge model configured, or the endpoint could not be reached."""


def build_judge() -> LLM:
    """Instantiate the judge client. Raises JudgeUnavailable when JUDGE_MODEL is unset."""
    if not SETTINGS.judge_configured:
        raise JudgeUnavailable(
            "No judge model set. Run `ollama list`, then put the exact tag in the [judge] "
            "section of config/settings.toml as model = \"<tag>\"."
        )
    return LLM(
        model=SETTINGS.judge_model,
        base_url=SETTINGS.judge_base_url,
        api_key=SETTINGS.judge_api_key,
        timeout_s=SETTINGS.judge_timeout_s,
        max_concurrency=SETTINGS.judge_max_concurrency,
        temperature=SETTINGS.judge_temperature,
        label="judge",
    )


async def judge_available(judge: LLM) -> bool:
    """Cheap liveness probe -- a local model that is not running should skip, not crash."""
    try:
        await judge.structured(
            JudgeQuestionRating,
            build_system("judge_question", SETTINGS.judge_strategy),
            render(
                user_template("judge_question"),
                role="Backend Engineer", seniority="mid", difficulty="medium",
                topic="Warm-up", requirements="Python",
                question="How do you profile a slow Python endpoint?", qtype="practical",
            ),
            stage="evaluator",
            max_retries=1,
        )
        return True
    except LLMUnavailable:
        return False


async def score_answer(
    judge: LLM, *, role: str, seniority: str, topic: str, question: str, rubric: str, answer: str
) -> JudgeAnswerScore:
    """Re-score one answer independently. The primary score is deliberately not shown."""
    return await judge.structured(
        JudgeAnswerScore,
        build_system("judge_answer", SETTINGS.judge_strategy),
        render(
            user_template("judge_answer"),
            role=role, seniority=seniority, topic=topic,
            question=question, rubric=rubric or "(no rubric supplied; judge on merit)",
            answer=answer,
        ),
        stage="evaluator",
    )


async def rate_question(
    judge: LLM, analysis: JobAnalysis, question: Question, difficulty: str
) -> JudgeQuestionRating:
    """Rate one question on grounding, realism, single-ask and difficulty fit."""
    return await judge.structured(
        JudgeQuestionRating,
        build_system("judge_question", SETTINGS.judge_strategy),
        render(
            user_template("judge_question"),
            role=analysis.role,
            seniority=analysis.seniority,
            difficulty=difficulty,
            topic=question.topic or "(unlabelled)",
            requirements=", ".join(r.name for r in analysis.requirements) or "(none extracted)",
            question=question.text,
            qtype=question.type,
        ),
        stage="questioner",
    )


async def rate_questions(
    judge: LLM, analysis: JobAnalysis, questions: list[Question], difficulty: str, limit: int = 6
) -> list[JudgeQuestionRating]:
    """Rate a sample of questions concurrently. Local models are slow, hence the cap."""
    sample = questions[:limit]
    results = await asyncio.gather(
        *(rate_question(judge, analysis, q, difficulty) for q in sample),
        return_exceptions=True,
    )
    return [r for r in results if isinstance(r, JudgeQuestionRating)]
