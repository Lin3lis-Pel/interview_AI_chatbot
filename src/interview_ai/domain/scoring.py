"""Everything that grades an answer: per-answer evaluation, the final report, and the
independent judge's smaller verdicts.

The judge models are filled by a SECOND model on a separate endpoint (eval harness only)
and are kept deliberately smaller than `Evaluation`: the judge scores and explains, it
does not drive the interview.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

Verdict = Literal["strong", "partial", "vague", "wrong", "no_answer"]
NextAction = Literal["advance", "drill_deeper", "clarify", "probe_technology", "raise_difficulty"]


class Scores(BaseModel):
    correctness: int = Field(ge=1, le=5)
    depth: int = Field(ge=1, le=5)
    practical_experience: int = Field(ge=1, le=5)
    communication: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)

    @property
    def mean(self) -> float:
        return round(sum(self.model_dump().values()) / 5, 2)


class Evaluation(BaseModel):
    verdict: Verdict
    scores: Scores
    # Flow control runs on this number, not on `next_action`: the interview asks a follow-up
    # only while coverage sits below the difficulty's threshold, and stops as soon as it is
    # met. Scored over EVERYTHING said so far on the current question, so a follow-up that
    # adds real information raises it and one that merely repeats does not.
    information_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "0-1: how much of what the question's rubric requires the candidate has "
            "demonstrated so far in this exchange. 1.0 = the rubric is fully satisfied."
        ),
    )
    covered: list[str] = Field(default_factory=list, description="Points the candidate did demonstrate.")
    missing: list[str] = Field(default_factory=list, description="Points a good answer would have had.")
    mentioned_technologies: list[str] = Field(
        default_factory=list, description="Technologies the candidate volunteered, worth probing."
    )
    next_action: NextAction
    follow_up: str = Field(
        default="",
        description="The follow-up question to ask now. Empty string only when next_action == 'advance'.",
    )

    @model_validator(mode="after")
    def _follow_up_present(self) -> "Evaluation":
        if self.next_action != "advance" and not self.follow_up.strip():
            raise ValueError("next_action requires a non-empty follow_up question.")
        return self


class TopicScore(BaseModel):
    topic: str
    score: float = Field(ge=0, le=5)
    comment: str = ""


# List limits for the report. `unmet_requirements` matches the analyzer's 25-requirement cap:
# a short interview against a requirement-heavy ad can legitimately leave more than 10 unmet,
# and the old cap of 10 then rejected the whole report.
_REPORT_LIST_CAPS: dict[str, int] = {
    "strengths": 6, "weaknesses": 6, "unmet_requirements": 25,
    "struggled_questions": 6, "study_plan": 8,
}


class FinalReport(BaseModel):
    overall_score: float = Field(ge=0, le=100, description="0-100 overall interview score.")
    recommendation: Literal["strong_yes", "yes", "borderline", "no"]
    recommendation_reason: str
    per_topic: list[TopicScore] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list, max_length=_REPORT_LIST_CAPS["strengths"])
    weaknesses: list[str] = Field(default_factory=list, max_length=_REPORT_LIST_CAPS["weaknesses"])
    unmet_requirements: list[str] = Field(
        default_factory=list, max_length=_REPORT_LIST_CAPS["unmet_requirements"],
        description="Requirements from the ad the candidate failed to demonstrate.",
    )
    struggled_questions: list[str] = Field(
        default_factory=list, max_length=_REPORT_LIST_CAPS["struggled_questions"]
    )
    study_plan: list[str] = Field(
        default_factory=list, max_length=_REPORT_LIST_CAPS["study_plan"],
        description="Concrete, ordered things to study before the real interview.",
    )

    @field_validator(*_REPORT_LIST_CAPS, mode="before")
    @classmethod
    def _trim_to_cap(cls, value: object, info: ValidationInfo) -> object:
        """Keep the first N items instead of rejecting the report.

        One surplus weakness used to fail validation, burn every retry, and leave the
        candidate with no report at all. The lists are ordered by importance, so trimming
        the tail loses the least.
        """
        cap = _REPORT_LIST_CAPS[info.field_name]
        return value[:cap] if isinstance(value, list) else value


class JudgeAnswerScore(BaseModel):
    verdict: Verdict
    scores: Scores
    reasoning: str = Field(default="", description="One or two sentences: what decided the score.")


class JudgeQuestionRating(BaseModel):
    grounded: bool = Field(description="Tests something the job ad actually asks for.")
    realistic: bool = Field(description="A working engineer in this role would really ask it.")
    single_ask: bool = Field(description="Asks exactly one thing.")
    difficulty_fit: Literal["too_easy", "about_right", "too_hard"]
    comment: str = ""

    @property
    def quality(self) -> float:
        """Fraction of the three boolean checks passed."""
        return sum([self.grounded, self.realistic, self.single_ask]) / 3
