"""One interview question and the batch a single generation call returns."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

QuestionType = Literal[
    "conceptual", "practical", "scenario", "project_experience", "debugging", "architecture", "behavioral"
]


class Question(BaseModel):
    topic: str = ""
    type: QuestionType = "conceptual"
    text: str
    targets: list[str] = Field(
        default_factory=list, description="Requirement names from the ad this question probes."
    )
    what_good_looks_like: str = Field(
        default="", description="Internal-only rubric. Never shown to the candidate mid-interview."
    )

    @field_validator("text")
    @classmethod
    def _non_trivial(cls, v: str) -> str:
        if len(v.strip()) < 15:
            raise ValueError("Question text is too short to be a real interview question.")
        return v.strip()


class QuestionBatch(BaseModel):
    questions: list[Question] = Field(min_length=1, max_length=6)
