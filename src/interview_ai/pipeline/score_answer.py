"""Stage 4-5: score one answer and decide what the interview does next.

Split out of the state machine on purpose. `InterviewSession` owns the bookkeeping --
which question we are on, what has been said, what each question established -- and this
module owns the single model call plus the flow-control rule that reads its result.
`InterviewSession.submit_answer` delegates here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import DIFFICULTY_RUBRIC
from ..domain.interview import Turn
from ..domain.scoring import Evaluation
from ..prompts import build_system, render, user_template
from ..utils.security import guard_output

if TYPE_CHECKING:
    from ..domain.interview import InterviewSession
    from ..llm import LLM


async def score_answer(session: "InterviewSession", llm: "LLM", answer: str) -> Evaluation:
    """Score the answer privately and advance the machine. Returns the evaluation."""
    q = session.current_question
    if q is None or session.finished:
        raise RuntimeError("The interview is already over.")

    asked = session.pending_followup or q.text
    is_followup = bool(session.pending_followup)

    evaluation = await llm.structured(
        Evaluation,
        build_system(
            "evaluator",
            session.strategy or session.settings.strategy_for("evaluator"),
            persona=session.persona or session.settings.persona,
            max_followups=session.max_followups,
        ),
        render(
            user_template("evaluator"),
            role=session.analysis.role,
            seniority=session.analysis.seniority,
            difficulty=f"{session.difficulty} -- {DIFFICULTY_RUBRIC[session.difficulty]}",
            topic=q.topic,
            # The type was assigned by code, so this classification is reliable.
            question_kind=(
                "SOFT SKILLS -- personality and working style; there is no correct answer"
                if q.type == "behavioral" else
                "TECHNICAL -- knowledge and experience"
            ),
            followups_used=session.followups_used,
            max_followups=session.max_followups,
            coverage_threshold=f"{session.coverage_threshold:.2f}",
            question=asked,
            rubric=q.what_good_looks_like or "(no rubric supplied; judge on merit)",
            exchange=session.exchange_for_current(),
            answer=answer,
        ),
        stage="evaluator",
    )
    evaluation.follow_up = guard_output(evaluation.follow_up)

    session.turns.append(
        Turn(question=asked, answer=answer, topic=q.topic,
             is_followup=is_followup, evaluation=evaluation)
    )

    # ---- flow control: arithmetic, not the model's mood ----------------
    # `next_action` still shapes WHAT a follow-up asks (drill the gap, clarify a wrong
    # claim, probe a volunteered technology). It no longer decides WHETHER to ask one:
    # left to itself the model chose `advance` in 0 of 48 recorded evaluations, so every
    # answer was drilled to the budget regardless of how good it was.
    #
    # Coverage decides instead. Reaching the threshold ends the question at once -- a
    # good answer earns the move on. Falling short buys a follow-up aimed at the gap,
    # and coverage is re-scored over the whole exchange, so a follow-up that adds real
    # information raises it while one that only repeats does not.
    covered_enough = evaluation.information_coverage >= session.coverage_threshold
    out_of_followups = session.followups_used >= session.max_followups

    if covered_enough or out_of_followups or not evaluation.follow_up:
        session.close_question(q, evaluation)
    else:
        session.pending_followup = evaluation.follow_up
        session.followups_used += 1

    return evaluation
