"""The interview plan and the interview state machine (stages 2 and 4-6 of the flow).

One `InterviewSession` holds the whole run: the question bank, the live exchange, and the
private per-answer scores. The UI only ever calls `current_prompt()` and `submit_answer()`.

Scores are recorded but deliberately never returned to the UI mid-interview -- the candidate
should not be able to read their own scoreboard while answering.

Everything here is state and arithmetic. The one model call the machine needs lives in
`interview_ai.pipeline.score_answer`, so this module has no import-time dependency on the
LLM stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from ..config import SETTINGS, Difficulty, Settings
from ..prompts import Strategy
from ..utils.security import redact_pii
from .job import JobAnalysis
from .question import Question
from .scoring import Evaluation

if TYPE_CHECKING:                       # import cycle: the LLM stack imports the domain
    from ..llm import LLM


class Topic(BaseModel):
    name: str
    weight: int = Field(ge=1, le=60, description="Integer percent of the interview.")
    rationale: str = Field(description="Why the ad justifies this weight.")
    grounded_in: list[str] = Field(
        default_factory=list, description="Requirement names from the analysis this topic covers."
    )


class InterviewPlan(BaseModel):
    topics: list[Topic] = Field(min_length=3, max_length=8)

    @model_validator(mode="after")
    def _weights_sum_to_100(self) -> "InterviewPlan":
        total = sum(t.weight for t in self.topics)
        if total != 100:
            raise ValueError(f"Topic weights must total exactly 100, got {total}. Adjust and resend.")
        return self

    def question_allocation(self, n: int) -> dict[str, int]:
        """Largest-remainder apportionment of `n` questions across weights."""
        raw = {t.name: t.weight * n / 100 for t in self.topics}
        base = {k: int(v) for k, v in raw.items()}
        left = n - sum(base.values())
        for name, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
            if left <= 0:
                break
            base[name] += 1
            left -= 1
        return {k: v for k, v in base.items() if v > 0}


@dataclass
class Turn:
    """One question (or follow-up) and the answer it received."""
    question: str
    answer: str
    topic: str
    is_followup: bool = False
    evaluation: Evaluation | None = None

    @property
    def score(self) -> float:
        return self.evaluation.scores.mean if self.evaluation else 0.0

    @property
    def coverage(self) -> float:
        return self.evaluation.information_coverage if self.evaluation else 0.0


@dataclass
class QuestionOutcome:
    """What one base question established, once its exchange closed.

    `met` is the interview's actual finding: did the candidate demonstrate what this
    question's requirements asked for? It is computed from coverage against the
    difficulty's threshold -- not inferred later by the reporting model -- which is what
    makes `unmet_requirements` in the final report evidence rather than a guess.
    """
    question: str
    topic: str
    targets: list[str]
    coverage: float
    threshold: float
    answers_given: int
    missing: list[str] = field(default_factory=list)

    @property
    def met(self) -> bool:
        return self.coverage >= self.threshold


@dataclass
class InterviewSession:
    analysis: JobAnalysis
    plan: InterviewPlan
    questions: list[Question]
    difficulty: Difficulty = "medium"
    strategy: Strategy | None = None     # None -> Settings.strategy_for('evaluator')
    persona: str | None = None           # None -> Settings.persona
    verbosity: str | None = None         # None -> Settings.verbosity
    # The session carries its own Settings so the UI's sliders (threshold, follow-up cap,
    # token caps) apply to THIS interview without mutating the global defaults.
    settings: Settings = SETTINGS

    index: int = 0                      # which base question we are on
    followups_used: int = 0              # follow-ups spent on the current base question
    pending_followup: str = ""           # set when the last evaluation asked for one
    turns: list[Turn] = field(default_factory=list)
    outcomes: list[QuestionOutcome] = field(default_factory=list)
    finished: bool = False

    @property
    def coverage_threshold(self) -> float:
        """How complete an answer must be before the interview moves on."""
        return self.settings.coverage_threshold_for(self.difficulty)

    @property
    def max_followups(self) -> int:
        return self.settings.max_followups_per_question

    # ------------------------------------------------------------------ state
    @property
    def current_question(self) -> Question | None:
        return self.questions[self.index] if self.index < len(self.questions) else None

    def current_prompt(self) -> str | None:
        """What to show the candidate right now: a pending follow-up, else the next question."""
        if self.finished:
            return None
        if self.pending_followup:
            return self.pending_followup
        q = self.current_question
        return q.text if q else None

    @property
    def progress(self) -> tuple[int, int]:
        return min(self.index + 1, len(self.questions)), len(self.questions)

    def exchange_for_current(self) -> str:
        """Turns already spent on the current base question, for follow-up context.

        Sliced by count rather than filtered by topic: several questions can share a
        topic, and coverage must be re-scored over THIS question's exchange only.
        `followups_used` is exactly how many answers this question has already taken.
        """
        spent = self.turns[-self.followups_used:] if self.followups_used else []
        lines = [f"Q: {t.question}\nA: {t.answer}" for t in spent]
        return "\n\n".join(lines) or "(this is the first exchange on this question)"

    # ------------------------------------------------------------------ main step
    async def submit_answer(self, llm: "LLM", answer: str) -> Evaluation:
        """Score the answer privately and advance the machine. Returns the evaluation.

        The model call and the flow-control rules live in
        `interview_ai.pipeline.score_answer`; imported inside the method so the domain
        layer stays importable without the LLM stack.
        """
        from ..pipeline.score_answer import score_answer

        return await score_answer(self, llm, answer)

    def close_question(self, question: Question, evaluation: Evaluation) -> None:
        """Record what this question established, then move to the next one."""
        self.outcomes.append(
            QuestionOutcome(
                question=question.text,
                topic=question.topic,
                targets=list(question.targets),
                coverage=evaluation.information_coverage,
                threshold=self.coverage_threshold,
                answers_given=self.followups_used + 1,
                missing=list(evaluation.missing),
            )
        )
        self._advance()

    def _advance(self) -> None:
        self.pending_followup = ""
        self.followups_used = 0
        self.index += 1
        if self.index >= len(self.questions):
            self.finished = True

    # ------------------------------------------------------------------ findings
    def unmet_requirements(self) -> list[str]:
        """Requirements whose questions never reached the coverage threshold.

        Computed from the exchange, so the report states what the interview actually
        measured instead of asking the model to remember it.
        """
        met = {t for o in self.outcomes if o.met for t in o.targets}
        unmet: list[str] = []
        for o in self.outcomes:
            if o.met:
                continue
            for target in o.targets:
                if target not in met and target not in unmet:
                    unmet.append(target)
        return unmet

    def coverage_summary(self) -> str:
        """Per-question coverage, for the reporting prompt."""
        if not self.outcomes:
            return "(no questions were completed)"
        lines = []
        for o in self.outcomes:
            verdict = "MET" if o.met else "NOT MET"
            gaps = f" | still missing: {', '.join(o.missing)}" if o.missing and not o.met else ""
            lines.append(
                f"- [{o.topic}] coverage {o.coverage:.2f} vs threshold {o.threshold:.2f} "
                f"-> {verdict} after {o.answers_given} answer(s); "
                f"targets: {', '.join(o.targets) or '(none recorded)'}{gaps}"
            )
        return "\n".join(lines)

    def skip(self) -> None:
        """Candidate passes on the question; recorded as a no-answer turn."""
        q = self.current_question
        if q is None:
            return
        self.turns.append(
            Turn(question=self.pending_followup or q.text, answer="(skipped)", topic=q.topic,
                 is_followup=bool(self.pending_followup))
        )
        # A skipped question demonstrated nothing: coverage 0, so its requirements are
        # recorded as unmet rather than quietly vanishing from the report.
        self.outcomes.append(
            QuestionOutcome(
                question=q.text, topic=q.topic, targets=list(q.targets),
                coverage=0.0, threshold=self.coverage_threshold,
                answers_given=self.followups_used, missing=["No answer was given."],
            )
        )
        self._advance()

    def end_early(self) -> None:
        self.finished = True

    # ------------------------------------------------------------------ views
    def topic_scores(self) -> dict[str, float]:
        by_topic: dict[str, list[float]] = {}
        for t in self.turns:
            if t.evaluation:
                by_topic.setdefault(t.topic, []).append(t.score)
        return {k: round(sum(v) / len(v), 2) for k, v in by_topic.items() if v}

    def transcript_for_model(self) -> str:
        """Full transcript with the private scores, for the report stage only."""
        blocks = []
        for i, t in enumerate(self.turns, 1):
            e = t.evaluation
            scored = (
                f"[scores: correctness {e.scores.correctness}, depth {e.scores.depth}, "
                f"practical {e.scores.practical_experience}, communication {e.scores.communication}, "
                f"relevance {e.scores.relevance} | verdict {e.verdict} | "
                f"missing: {', '.join(e.missing) or 'nothing'}]"
                if e else "[not scored -- skipped]"
            )
            tag = "follow-up" if t.is_followup else "question"
            blocks.append(
                f"--- {i}. [{t.topic}] ({tag})\nQ: {t.question}\nA: {t.answer}\n{scored}"
            )
        return "\n".join(blocks) or "(no answers were given)"

    def transcript_for_export(self) -> str:
        """Human-readable, PII-redacted, no scores. Safe to download or share."""
        lines = [f"# Mock interview -- {self.analysis.title or self.analysis.role}", ""]
        for i, t in enumerate(self.turns, 1):
            prefix = "Follow-up" if t.is_followup else f"Q{i}"
            lines += [f"## {prefix} [{t.topic}]", t.question, "", "**Answer**", redact_pii(t.answer), ""]
        return "\n".join(lines)
