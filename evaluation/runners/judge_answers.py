"""Feed SAVED answers to the local judge -- zero paid API calls.

The two-phase flow this enables
-------------------------------
  Phase 1 (costs money):  uv run python -m evaluation.runners.benchmark
      The primary model analyses ads, writes questions and scores answers. Every scored
      answer is appended to artifacts/artifacts-<stamp>.jsonl as a numbered,
      self-contained record: question, rubric, the candidate answer, and the primary
      model's scores all ride along in the same line.

  Phase 2 (free, local):  uv run python -m evaluation.runners.judge_answers
      This script. Reads those records back, sends each answer to the local judge model
      (Ollama), and compares the judge's scores against the primary's. The primary model
      is never called -- you can rerun this as often as you like, e.g. after changing the
      judge prompt or swapping the judge model, without spending a cent.

Usage
-----
    uv run python -m evaluation.runners.judge_answers                     # newest artifacts file
    uv run python -m evaluation.runners.judge_answers --artifacts artifacts/artifacts-XXX.jsonl
    uv run python -m evaluation.runners.judge_answers --strategy few_shot # one strategy's records only
    uv run python -m evaluation.runners.judge_answers --limit 5           # quick smoke test

Writes artifacts/judge-<stamp>.md / .json, plus judge-<stamp>.jsonl with every raw
judge score (numbered, referencing the source record's `n`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from interview_ai.config import SETTINGS
from interview_ai.llm import LLM, LLMUnavailable
from interview_ai.domain import Evaluation, JudgeAnswerScore
from evaluation import metrics as M
from evaluation import stats as S
from evaluation.artifacts import ArtifactLog, latest_artifacts, load_artifacts
from evaluation.fixtures.answers import ADVERSARIAL_ANSWERS, ANSWERS
from evaluation.fixtures.job_ads import CLEAN_ADS
from evaluation.judges.gemma import JudgeUnavailable, build_judge, judge_available, score_answer

REPORTS = Path(__file__).resolve().parents[2] / "artifacts"

# Fallback for artifacts written before evaluation records became self-contained: the
# scripted answers live in fixtures, keyed by the same fixture id the record carries.
_FIXTURES = {f.id: f for f in ANSWERS + ADVERSARIAL_ANSWERS}


@dataclass
class Task:
    """One saved answer to re-judge: the source record plus everything the judge needs."""
    n: int                 # record number in the source artifacts file
    fixture: str
    strategy: str
    tier: str
    primary: Evaluation    # what the primary model said, never shown to the judge
    role: str
    seniority: str
    topic: str
    question: str
    rubric: str
    answer: str


def tasks_from(records: list[dict], strategy: str | None, limit: int | None) -> list[Task]:
    """Rebuild judge tasks from saved `evaluation` records, fixtures filling any gaps."""
    default_role = CLEAN_ADS[0].role
    default_seniority = CLEAN_ADS[0].seniority

    tasks: list[Task] = []
    for r in records:
        if r.get("stage") != "evaluation":
            continue
        if strategy and r.get("strategy") != strategy:
            continue
        meta = r.get("meta", {})
        fx = _FIXTURES.get(r.get("fixture", ""))
        try:
            primary = Evaluation.model_validate(r["payload"])
        except Exception:
            continue  # a malformed line from an interrupted run
        question = meta.get("question") or (fx.question if fx else "")
        answer = meta.get("answer") or (fx.answer if fx else "")
        if not question or not answer:
            continue  # nothing to judge without the exchange itself
        tasks.append(Task(
            n=r.get("n", 0),
            fixture=r.get("fixture", "?"),
            strategy=r.get("strategy", "?"),
            tier=meta.get("tier", fx.tier if fx else "?"),
            primary=primary,
            role=meta.get("role", default_role),
            seniority=meta.get("seniority", default_seniority),
            topic=meta.get("topic") or (fx.topic if fx else ""),
            question=question,
            rubric=meta.get("rubric") or (fx.rubric if fx else ""),
            answer=answer,
        ))
    if limit:
        tasks = tasks[:limit]
    return tasks


async def judge_all(judge: LLM, tasks: list[Task]) -> list[tuple[Task, JudgeAnswerScore | None]]:
    async def one(t: Task):
        try:
            return t, await score_answer(
                judge, role=t.role, seniority=t.seniority, topic=t.topic,
                question=t.question, rubric=t.rubric, answer=t.answer,
            )
        except LLMUnavailable as exc:
            print(f"  [n={t.n} {t.fixture}] judge failed: {exc}")
            return t, None

    return await asyncio.gather(*(one(t) for t in tasks))


def render_markdown(payload: dict) -> str:
    lines = [
        "# interview_AI - offline judge report",
        "",
        f"- **Run:** {payload['timestamp']}",
        f"- **Source answers:** `{Path(payload['artifacts']).name}` "
        f"({payload['n_judged']} of {payload['n_available']} records judged)",
        f"- **Judge:** `{payload['judge_model']}` at {payload['judge_base_url']} "
        f"(strategy: {payload['judge_strategy']}, temperature: {payload['judge_temperature']})",
        f"- **Paid API calls made: 0** -- every answer came from the saved artifacts file.",
        f"- **Wall time:** {payload['wall_time_s']:.1f}s",
        "",
        "## Agreement per strategy",
        "",
        "`bias` is signed: positive means the primary model scored its own generations "
        "higher than the judge -- the self-preference signature.",
        "",
        "| Strategy | n | score agreement | verdict agreement | mean bias | kappa | rho | MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in payload["per_strategy"].items():
        a = block["agreement"]
        lines.append(
            f"| {name} | {block['n']} | {block['score_agreement']:.3f} | "
            f"{block['verdict_agreement']:.3f} | {block['mean_bias']:+.3f} | "
            f"{a.get('cohen_kappa', float('nan')):.3f} | "
            f"{a.get('spearman_rho', float('nan')):.3f} | {a.get('mae', float('nan')):.3f} |"
        )

    lines += [
        "",
        "## Every answer, side by side",
        "",
        "`#` is the record number in the source artifacts file.",
        "",
        "| # | strategy | answer | gold tier | primary | judge | delta | primary verdict | judge verdict |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in payload["answers"]:
        lines.append(
            f"| {row['n']} | {row['strategy']} | {row['fixture']} | {row['tier']} | "
            f"{row['primary_mean']:.1f} | {row['judge_mean']:.1f} | {row['delta']:+.1f} | "
            f"{row['primary_verdict']} | {row['judge_verdict']} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "- The judge never sees the primary score -- each answer is re-scored blind, so the "
        "comparison is a genuine second opinion, not anchoring.",
        "- A consistently positive `delta`/`bias` means the primary model marks its own "
        "homework generously. The gold tiers in evaluation/fixtures/answers.py stay the ground truth.",
        "- Rerun after changing the judge prompt (prompts/judge_answer/) or the judge model "
        "in config/settings.toml -- it costs nothing.",
    ]
    return "\n".join(lines)


async def main() -> int:
    # cp1252 console guard -- report files are always UTF-8, only the echo needs this.
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Re-score saved answers with the local judge")
    parser.add_argument("--artifacts", help="artifacts-*.jsonl to read (default: newest)")
    parser.add_argument("--strategy", help="only judge records from this strategy")
    parser.add_argument("--limit", type=int, help="judge at most N records")
    args = parser.parse_args()

    path = Path(args.artifacts) if args.artifacts else latest_artifacts(REPORTS)
    if not path or not Path(path).exists():
        print("No artifacts file found. Run `uv run python -m evaluation.runners.benchmark` first -- "
              "it saves every scored answer to artifacts/artifacts-<stamp>.jsonl.")
        return 2

    records = load_artifacts(path)
    available = sum(1 for r in records if r.get("stage") == "evaluation")
    tasks = tasks_from(records, args.strategy, args.limit)
    if not tasks:
        print(f"{Path(path).name} contains no judgeable `evaluation` records"
              + (f" for strategy {args.strategy!r}." if args.strategy else "."))
        return 2
    print(f"{len(tasks)} saved answers from {Path(path).name} -> judge, no paid API calls.")

    try:
        judge = build_judge()
    except JudgeUnavailable as exc:
        print(exc)
        return 2
    if not await judge_available(judge):
        print(f"Judge {SETTINGS.judge_model} did not respond at {SETTINGS.judge_base_url}. "
              "Is `ollama serve` running?")
        await judge.aclose()
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = ArtifactLog(REPORTS / f"judge-{stamp}.jsonl", run_id=stamp)

    started = time.perf_counter()
    results = await judge_all(judge, tasks)
    judge_usage = judge.usage.summary()
    await judge.aclose()

    # ---- aggregate ----
    answers: list[dict] = []
    by_strategy: dict[str, S.Collected] = {}
    pairwise: dict[str, dict[str, list[float]]] = {}
    for t, jr in results:
        if jr is None:
            continue
        log.write(strategy=t.strategy, stage="judge_answer_score", fixture=t.fixture,
                  payload=jr, source_n=t.n, source_run=str(Path(path).name), tier=t.tier,
                  judge_model=SETTINGS.judge_model,
                  primary_mean=t.primary.scores.mean, primary_verdict=t.primary.verdict)
        answers.append({
            "n": t.n, "strategy": t.strategy, "fixture": t.fixture, "tier": t.tier,
            "primary_mean": t.primary.scores.mean, "judge_mean": jr.scores.mean,
            "delta": round(t.primary.scores.mean - jr.scores.mean, 2),
            "primary_verdict": t.primary.verdict, "judge_verdict": jr.verdict,
        })
        obs = by_strategy.setdefault(t.strategy, S.Collected())
        obs.primary_scores.append(t.primary.scores.mean)
        obs.judge_scores.append(jr.scores.mean)
        obs.primary_verdicts.append(t.primary.verdict)
        obs.judge_verdicts.append(jr.verdict)
        pw = pairwise.setdefault(t.strategy, {"score": [], "verdict": []})
        pw["score"].append(M.judge_score_agreement(t.primary, jr))
        pw["verdict"].append(M.judge_verdict_agreement(t.primary, jr))

    if not answers:
        print("The judge failed on every record -- nothing to report.")
        return 1

    per_strategy = {}
    for name, obs in sorted(by_strategy.items()):
        agreement = obs.summary().get("judge_agreement", {})
        pw = pairwise[name]
        per_strategy[name] = {
            "n": len(obs.judge_scores),
            "score_agreement": round(sum(pw["score"]) / len(pw["score"]), 3),
            "verdict_agreement": round(sum(pw["verdict"]) / len(pw["verdict"]), 3),
            "mean_bias": agreement.get("mean_bias", 0.0),
            "agreement": agreement,
        }

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifacts": str(path),
        "n_available": available,
        "n_judged": len(answers),
        "judge_model": SETTINGS.judge_model,
        "judge_base_url": SETTINGS.judge_base_url,
        "judge_strategy": SETTINGS.judge_strategy,
        "judge_temperature": SETTINGS.judge_temperature,
        "judge_usage": judge_usage,
        "wall_time_s": time.perf_counter() - started,
        "per_strategy": per_strategy,
        "answers": sorted(answers, key=lambda a: a["n"]),
    }

    markdown = render_markdown(payload)
    (REPORTS / f"judge-{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS / f"judge-{stamp}.md").write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"\nWritten to {REPORTS / f'judge-{stamp}.md'}")
    print(f"{log.count} raw judge scores in {log.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
