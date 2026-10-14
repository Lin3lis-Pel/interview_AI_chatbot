"""Prompt loading and assembly. The prompt TEXT lives in the `prompts/` directory.

Where the words are
-------------------
Every piece of prompt text is a plain file in this package, under `<stage>/`:

    system.txt       the rules -- shared by ALL five strategies
    user.txt         the user-message template ({placeholders} filled at call time)
    guide.txt        the numbered procedure appended by the `guided` strategy
    critique.txt     the draft-audit-revise checklist appended by `self_critique`
    examples/        worked examples appended by `one_shot` (first file) and
                     `few_shot` (first three files)

`_shared/` holds the anti-injection and anti-invention lines; the token
[[ANTI_INJECTION]] / [[NO_INVENTION]] in a system.txt is replaced by that shared text at
load time, so the security wording cannot drift apart between stages.

`VERSION.txt` is the prompt version stamped into every eval report -- bump it
whenever you edit any prompt file, then diff two reports.

Edit a file, restart the app (or rerun the eval), done. To see the fully assembled prompt
for any stage x strategy combination, run:  uv run python -m interview_ai.prompts
which writes them to `artifacts/prompts/`.

Five prompting strategies, comparable by construction
------------------------------------------------------
  zero_shot     rules only (the baseline)
  one_shot      rules + ONE worked example
  few_shot      rules + THREE worked examples
  guided        rules + an explicit reasoning procedure to follow before answering
  self_critique rules + a stage-specific draft-audit-revise checklist

The **rules are byte-identical across all five**. Only the appended block differs, so a
metric difference between strategies is caused by the technique and nothing else -- an
ablation where the arms differ in two ways measures nothing. Enforced by
tests/integration/test_strategies.py::test_rules_are_identical_across_strategies.

Assembly is `build_system(stage, strategy)`. Format placeholders are substituted into the
base rules only; exemplar and guide blocks are concatenated afterwards and never passed
through `str.format`, because exemplar JSON contains braces.

Exemplar hygiene
----------------
Worked examples deliberately use roles that do NOT appear in the eval fixtures
(QA / Automation, Cybersecurity, Data Scientist). If the model copies an exemplar instead of
reading the input, the output carries the exemplar's role and subjects -- which the
`exemplar_leakage` metric detects. Few-shot prompting on a small model buys format
compliance and costs independence; this setup measures both sides of that trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PROMPTS_DIR = Path(__file__).resolve().parent

Strategy = Literal["zero_shot", "one_shot", "few_shot", "guided", "self_critique"]
STRATEGIES: tuple[Strategy, ...] = (
    "zero_shot", "one_shot", "few_shot", "guided", "self_critique",
)

StageKey = Literal["analyzer", "planner", "questioner", "evaluator", "reporter"]

# Stages that make up the product pipeline, and can therefore be swept by strategy.
PRODUCT_STAGES: tuple[str, ...] = ("analyzer", "planner", "questioner", "evaluator", "reporter")
# Judge stages get a `guided` procedure but deliberately NO exemplars: a worked example
# shared with the primary model would correlate the two scorers, and a correlated second
# opinion measures nothing.
JUDGE_STAGES: tuple[str, ...] = ("judge_answer", "judge_question")


# --------------------------------------------------------------------------- loading
def _read(rel: str) -> str:
    path = PROMPTS_DIR / rel
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file missing: {path}. The prompt text ships inside the "
            "interview_ai.prompts package -- reinstall with `uv sync` if it is missing."
        )
    return path.read_text(encoding="utf-8")


_ANTI_INJECTION = _read("_shared/anti_injection.txt")
_NO_INVENTION = _read("_shared/no_invention.txt")

_SHARED_TOKENS = {
    "[[ANTI_INJECTION]]": _ANTI_INJECTION,
    "[[NO_INVENTION]]": _NO_INVENTION,
}

PROMPT_VERSION = _read("VERSION.txt").strip()


def _load(rel: str) -> str:
    """Read one prompt file and splice in the shared security/grounding lines."""
    text = _read(rel)
    for token, content in _SHARED_TOKENS.items():
        text = text.replace(token, content)
    return text


# Interviewer personas. Tone only -- they are appended AFTER the rules and never replace
# them, so the syllabus, the rubric and the scoring bar are identical whichever is chosen.
# Applied to the stages the candidate actually hears (questions and follow-ups); the
# analyzer, planner and reporter stay neutral, because a "friendly" hiring report that
# softens a gap is exactly the report this project exists not to write.
PERSONAS: tuple[str, ...] = ("neutral", "friendly", "strict")
PERSONA_STAGES: tuple[str, ...] = ("questioner", "evaluator")


def persona_block(persona: str) -> str:
    """The tone instructions for one persona, or '' for the neutral default."""
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona {persona!r}; expected one of {PERSONAS}.")
    return _read(f"personas/{persona}.txt")


# Response length. Like personas, appended after the rules and only for the stages whose
# text the candidate reads at length (the questions and the final report). The scorer is
# left alone so a length preference cannot move a score. `standard` adds nothing, so the
# default keeps every strategy arm byte-identical.
VERBOSITY: tuple[str, ...] = ("standard", "concise", "detailed")
VERBOSITY_STAGES: tuple[str, ...] = ("questioner", "reporter")


def verbosity_block(verbosity: str) -> str:
    """The length instructions for one verbosity level, or '' for the standard default."""
    if verbosity not in VERBOSITY:
        raise ValueError(f"Unknown verbosity {verbosity!r}; expected one of {VERBOSITY}.")
    return "" if verbosity == "standard" else _read(f"verbosity/{verbosity}.txt")


def _examples(stage: str) -> list[str]:
    """Worked examples in filename order: example_1_* is the one_shot example."""
    examples_dir = PROMPTS_DIR / stage / "examples"
    return [_load(f"{stage}/examples/{path.name}")
            for path in sorted(examples_dir.glob("example_*.txt"))]


# --------------------------------------------------------------------------- assembly
# Prepended to every exemplar block. Without this, a small model treats the example as a
# template to fill in rather than a shape to imitate.
_EXEMPLAR_WARNING = """\
--------------------------------------------------------------------------------
WORKED EXAMPLE{plural} -- SHAPE REFERENCE ONLY

{count} example{plural} follow{verb}, drawn from {a}DIFFERENT job{plural2} than the one you are about
to handle. Use {them} to see the level of specificity expected, nothing more.

Do NOT copy any subject name, weight, requirement, technology, phrasing or score from
{a2}example into your answer. An answer that reuses example content describes the wrong job
and is worse than no answer. Everything you output must come from the input you are given.
--------------------------------------------------------------------------------
"""


def _exemplar_header(count: int) -> str:
    if count == 1:
        return _EXEMPLAR_WARNING.format(
            plural="", count="One", verb="s", a="a ", plural2="", them="it", a2="the ",
        )
    return _EXEMPLAR_WARNING.format(
        plural="S", count=str(count), verb="", a="", plural2="s", them="them", a2="any ",
    )


def render(template: str, **kwargs: object) -> str:
    return template.format(**kwargs)


@dataclass(frozen=True)
class StageSpec:
    key: str
    system: str
    user: str
    guide: str
    critique: str
    exemplars: list[str] = field(default_factory=list)


def _spec(key: str, with_examples: bool = True) -> StageSpec:
    guide_path = PROMPTS_DIR / key / "guide.txt"
    critique_path = PROMPTS_DIR / key / "critique.txt"
    return StageSpec(
        key=key,
        system=_load(f"{key}/system.txt"),
        user=_load(f"{key}/user.txt"),
        guide=_load(f"{key}/guide.txt") if guide_path.exists() else "",
        critique=_load(f"{key}/critique.txt") if critique_path.exists() else "",
        exemplars=_examples(key) if with_examples else [],
    )


STAGES: dict[str, StageSpec] = {
    **{key: _spec(key) for key in PRODUCT_STAGES},
    **{key: _spec(key, with_examples=False) for key in JUDGE_STAGES},
}

_N_EXEMPLARS: dict[str, int] = {
    "zero_shot": 0, "one_shot": 1, "few_shot": 3, "guided": 0, "self_critique": 0,
}


def build_system(
    stage: str, strategy: Strategy = "zero_shot", persona: str = "neutral",
    verbosity: str = "standard", **fmt: object,
) -> str:
    """Assemble the system prompt for one stage under one strategy and persona.

    Only the base rules are `.format`ed -- exemplar blocks contain JSON braces and must be
    concatenated raw. The persona block is appended last and only for the stages the
    candidate hears, so it cannot disturb the strategy ablation: with the default persona
    every arm is byte-identical to before.
    """
    spec = STAGES[stage]
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}.")

    base = spec.system.format(**fmt) if fmt else spec.system
    if persona != "neutral" and stage in PERSONA_STAGES:
        base = base + "\n" + persona_block(persona)
    if verbosity != "standard" and stage in VERBOSITY_STAGES:
        base = base + "\n" + verbosity_block(verbosity)

    if strategy == "guided":
        # A stage with no guide block falls back to rules only, so a global strategy
        # setting cannot silently produce a broken prompt.
        return f"{base}\n{spec.guide}" if spec.guide else base

    if strategy == "self_critique":
        # The model returns only the corrected final object. The private draft and audit
        # add no response fields, keeping schemas identical to every other strategy arm.
        return f"{base}\n{spec.critique}" if spec.critique else base

    n = min(_N_EXEMPLARS[strategy], len(spec.exemplars))
    if n == 0:
        return base

    chosen = spec.exemplars[:n]
    blocks = "\n".join(
        f"--- EXAMPLE {i} OF {n} ---\n{ex}" for i, ex in enumerate(chosen, 1)
    )
    return f"{base}\n{_exemplar_header(n)}\n{blocks}"


def user_template(stage: str) -> str:
    return STAGES[stage].user


# Exemplar-specific vocabulary, exported so the eval harness can detect copying. If any of
# these appear in output produced for a fixture ad, the model reused the example.
# Update this list if you change roles or subjects in prompts/*/examples/example_* files.
EXEMPLAR_FINGERPRINTS: dict[str, list[str]] = {
    "roles": ["QA / Automation", "Cybersecurity", "Data Scientist"],
    "topics": [
        "Test automation strategy", "Flaky test diagnosis", "TypeScript for test code",
        "CI pipeline performance", "Collaboration with developers",
        "Threat modelling practice", "Web application vulnerabilities",
        "Secure SDLC programme design", "Vulnerability triage and bug bounty",
        "Influence and stakeholder communication",
        "Experiment design and A/B testing", "Statistics fundamentals", "Python for analysis",
        "Forecasting and past projects", "Explaining results to stakeholders",
    ],
    "terms": ["Playwright", "flaky test", "threat model", "OWASP", "A/B test", "k6",
              "bug bounty", "scikit-learn"],
}
