"""Write every fully-assembled prompt to artifacts/prompts/ for inspection.

    uv run python -m interview_ai.prompts

One file per stage x strategy, e.g. `analyzer.few_shot.txt` -- the EXACT system prompt the
model receives for that combination, with the user-message template appended for reference.
Read two side by side to see precisely what one_shot adds on top of zero_shot.

Generated output. To change a prompt, edit the source files in this package (see README.md).
"""
from __future__ import annotations

from pathlib import Path

from . import (
    JUDGE_STAGES,
    PRODUCT_STAGES,
    PROMPT_VERSION,
    STRATEGIES,
    build_system,
    user_template,
)

OUT = Path(__file__).resolve().parents[3] / "artifacts" / "prompts"

# Placeholders normally filled at call time; representative values so the files read naturally.
_FMT: dict[str, dict] = {"questioner": {"n": 3}, "evaluator": {"max_followups": 2}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.txt"):
        stale.unlink()

    written = 0
    for stage in (*PRODUCT_STAGES, *JUDGE_STAGES):
        # Judge stages carry no exemplars, so one_shot and few_shot would be byte-identical
        # copies of zero_shot. Only the two that differ are worth writing.
        strategies = ("zero_shot", "guided") if stage in JUDGE_STAGES else STRATEGIES
        for strategy in strategies:
            system = build_system(stage, strategy, **_FMT.get(stage, {}))
            (OUT / f"{stage}.{strategy}.txt").write_text(
                f"### prompt version {PROMPT_VERSION} | stage {stage} | strategy {strategy}\n"
                f"{'=' * 78}\n### SYSTEM\n{'=' * 78}\n{system}\n\n"
                f"{'=' * 78}\n### USER TEMPLATE\n{'=' * 78}\n{user_template(stage)}\n",
                encoding="utf-8",
            )
            written += 1
    print(f"{written} compiled prompts in {OUT}")


if __name__ == "__main__":
    main()
