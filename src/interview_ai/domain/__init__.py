"""The shapes the whole system agrees on.

Split by subject -- the job ad, the plan and the running interview, the questions, the
scoring -- but re-exported here as one namespace, so `from interview_ai.domain import
JobAnalysis, Question` keeps working wherever the model happens to live.

These pydantic models are the *only* shapes the LLM is allowed to return. `instructor`
validates every response against them and re-prompts on failure, so a malformed model
reply never reaches the UI.
"""
from __future__ import annotations

from .interview import (
    InterviewPlan,
    InterviewSession,
    QuestionOutcome,
    Topic,
    Turn,
)
from .job import Criticality, JobAnalysis, Requirement, Seniority
from .question import Question, QuestionBatch, QuestionType
from .scoring import (
    Evaluation,
    FinalReport,
    JudgeAnswerScore,
    JudgeQuestionRating,
    NextAction,
    Scores,
    TopicScore,
    Verdict,
)

__all__ = [
    "Criticality", "Evaluation", "FinalReport", "InterviewPlan", "InterviewSession",
    "JobAnalysis", "JudgeAnswerScore", "JudgeQuestionRating", "NextAction", "Question",
    "QuestionBatch", "QuestionOutcome", "QuestionType", "Requirement", "Scores",
    "Seniority", "Topic", "TopicScore", "Turn", "Verdict",
]
