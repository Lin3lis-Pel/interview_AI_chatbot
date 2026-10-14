"""Model-setting experiments: change one setting, hold everything else fixed, compare.

    uv run python -m evaluation.runners.experiments max-tokens   # planner output cap sweep
    uv run python -m evaluation.runners.experiments verbosity    # concise vs standard vs detailed

Both hit the real API (a few cents) over the four clean fixture ads, and neither needs the
independent judge: length, success rate, tokens and cost are measured by plain code, and the
existing code metrics check that nothing else moved. Results land in artifacts/.

max-tokens
    The planner is called with the same analysis under three output caps. On the gpt-5
    family reasoning tokens are billed against the same cap as the JSON, so a cap that looks
    generous for the visible output can still truncate it.

verbosity
    Questions: generated per ad under each length setting, then measured for length,
    grounding, rubric presence and duplication.
    Report: ONE fixed transcript per ad (standard questions, a simulated candidate, scored
    once) is reported on under each setting, so only the length instruction differs between
    arms. Overall score and unmet requirements should not move; only length should.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel
from tabulate import tabulate

from interview_ai.config import SETTINGS
from interview_ai.domain import InterviewSession, JobAnalysis
from interview_ai.llm import LLM, LLMUnavailable
from interview_ai.pipeline import analyze_job, build_report, generate_questions, plan_interview
from interview_ai.prompts import PROMPT_VERSION, VERBOSITY
from evaluation import metrics as M
from evaluation.fixtures.job_ads import CLEAN_ADS

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
PLANNER_CAPS = (1000, 2000, 5000)
REPEATS = 2
QUESTIONS = 6
TRANSCRIPT_TIERS = ("strong", "partial", "vague", "strong")


def words(*texts: str) -> int:
    return sum(len(t.split()) for t in texts if t)


def table(headers: list[str], rows: list[list]) -> str:
    return tabulate(rows, headers=headers, tablefmt="github", disable_numparse=True)


async def analyses(llm: LLM) -> list[tuple[str, JobAnalysis]]:
    """One shared analysis per ad, so every arm starts from identical input."""
    results = await asyncio.gather(*(analyze_job(llm, ad.text) for ad in CLEAN_ADS))
    return [(ad.id, a) for ad, a in zip(CLEAN_ADS, results)]


# --------------------------------------------------------------------------- max-tokens
async def planner_call(analysis: JobAnalysis, cap: int) -> dict:
    llm = LLM(settings=replace(SETTINGS, max_tokens_overrides={"planner": cap}))
    started = time.perf_counter()
    try:
        await plan_interview(llm, analysis, "medium")
        ok, error = True, ""
    except LLMUnavailable as exc:
        ok, error = False, str(exc)
    finally:
        await llm.aclose()
    u = llm.usage.summary()
    return {"cap": cap, "ok": ok, "error": error[:200],
            "truncated": any(k in error.lower() for k in ("length", "max_tokens", "incomplete")),
            "seconds": round(time.perf_counter() - started, 1),
            "completion_tokens": u["completion_tokens"], "cost_usd": u["cost_usd"]}


async def max_tokens() -> tuple[str, dict]:
    llm = LLM()
    ads = await analyses(llm)
    await llm.aclose()

    calls = [planner_call(a, cap) for cap in PLANNER_CAPS for _, a in ads for _ in range(REPEATS)]
    records = await asyncio.gather(*calls)

    rows = []
    for cap in PLANNER_CAPS:
        arm = [r for r in records if r["cap"] == cap]
        good = [r for r in arm if r["ok"]]
        rows.append([
            str(cap),
            f"{len(good)}/{len(arm)}",
            str(sum(r["truncated"] for r in arm)),
            f"{statistics.mean(r['completion_tokens'] for r in good):.0f}" if good else "-",
            f"{statistics.median(r['seconds'] for r in arm):.1f}s",
            f"${sum(r['cost_usd'] for r in arm):.4f}",
        ])
    md = table(["Planner max_tokens", "Valid plans", "Failed on length",
                "Mean output tokens (successes)", "Median time per call",
                "Recorded cost (successes only)"], rows)
    md += ("\n\nFailed calls are still billed by OpenRouter, but usage is only reported back "
           "on a successful response, so they cost more than this column shows. The median "
           "time includes instructor's re-prompts: a truncated call is retried before it fails.")
    return md, {"caps": PLANNER_CAPS, "repeats": REPEATS, "records": records}


# --------------------------------------------------------------------------- verbosity
class CandidateAnswer(BaseModel):
    answer: str


async def candidate_answer(llm: LLM, question: str, tier: str) -> str:
    brief = {
        "strong": "Answer with a specific example: the tool, a number, the trade-off. 4-5 sentences.",
        "partial": "Know the vocabulary but give no numbers, tools or real project. 3 sentences.",
        "vague": "Restate the question in general terms and never commit to a method. 2 sentences.",
    }[tier]
    reply = await llm.structured(
        CandidateAnswer,
        f"You are a job candidate in a technical interview. {brief} Never break character.",
        f"The interviewer asks:\n\n{question}",
    )
    return reply.answer


def question_lengths(questions) -> float:
    return statistics.mean(words(q.text) for q in questions) if questions else 0.0


def report_length(report) -> int:
    return words(report.recommendation_reason, *(t.comment for t in report.per_topic),
                 *report.strengths, *report.weaknesses, *report.unmet_requirements,
                 *report.study_plan)


async def verbosity_for_ad(llm: LLM, analysis: JobAnalysis) -> dict:
    plan = await plan_interview(llm, analysis, "medium")
    by_arm = dict(zip(VERBOSITY, await asyncio.gather(*(
        generate_questions(llm, analysis, plan, "medium", QUESTIONS, verbosity=v)
        for v in VERBOSITY
    ))))

    # One fixed transcript, built from the STANDARD questions and scored once. Every report
    # arm below receives a copy of exactly this session; only the length setting differs.
    no_followups = replace(SETTINGS, max_followups_per_question=0)
    base = InterviewSession(analysis=analysis, plan=plan,
                            questions=by_arm["standard"][:len(TRANSCRIPT_TIERS)],
                            difficulty="medium", settings=no_followups)
    for tier in TRANSCRIPT_TIERS:
        await base.submit_answer(llm, await candidate_answer(llm, base.current_prompt(), tier))

    # A report that fails schema validation is a result, not a crash: record it per arm.
    reports, errors = {}, {}
    for v in VERBOSITY:
        session = copy.deepcopy(base)
        session.verbosity = v
        try:
            reports[v] = await build_report(llm, session)
        except LLMUnavailable as exc:
            reports[v], errors[v] = None, str(exc)[:300]

    out = {}
    for v in VERBOSITY:
        r = reports[v]
        out[v] = {
            "question_words": question_lengths(by_arm[v]),
            "question_grounding": M.question_grounding(by_arm[v], analysis),
            "rubric_present": M.rubric_present(by_arm[v]),
            "duplicate_rate": M.duplicate_rate(by_arm[v]),
            "report_ok": r is not None,
            "report_error": errors.get(v, ""),
            "report_words": report_length(r) if r else None,
            "report_items": (len(r.strengths) + len(r.weaknesses) + len(r.study_plan)) if r else None,
            "overall_score": r.overall_score if r else None,
            "unmet": len(r.unmet_requirements) if r else None,
            "sample_question": by_arm[v][0].text if by_arm[v] else "",
        }
    return out


async def verbosity() -> tuple[str, dict]:
    llm = LLM()
    ads = await analyses(llm)
    per_ad = dict(zip(
        (ad_id for ad_id, _ in ads),
        await asyncio.gather(*(verbosity_for_ad(llm, a) for _, a in ads)),
    ))
    usage = llm.usage.summary()
    await llm.aclose()

    def mean(v: str, key: str, fmt: str) -> str:
        """Mean over the ads where the value exists; report metrics skip failed reports."""
        vals = [per_ad[ad][v][key] for ad in per_ad if per_ad[ad][v][key] is not None]
        return format(statistics.mean(vals), fmt) if vals else "-"

    rows = [[v,
             mean(v, "question_words", ".1f"), mean(v, "question_grounding", ".2f"),
             mean(v, "rubric_present", ".2f"), mean(v, "duplicate_rate", ".2f"),
             f"{sum(per_ad[ad][v]['report_ok'] for ad in per_ad)}/{len(per_ad)}",
             mean(v, "report_words", ".0f"), mean(v, "report_items", ".1f"),
             mean(v, "overall_score", ".1f"), mean(v, "unmet", ".1f")]
            for v in VERBOSITY]
    md = table(["Verbosity", "Words / question", "Question grounding", "Rubric present",
                "Duplicate rate", "Valid reports", "Report words", "Report list items",
                "Overall score", "Unmet requirements"], rows)
    failed = [f"- `{ad}` / {v}: {per_ad[ad][v]['report_error'][:160]}"
              for ad in per_ad for v in VERBOSITY if not per_ad[ad][v]["report_ok"]]
    if failed:
        md += "\n\nReports that failed schema validation:\n\n" + "\n".join(failed)
    samples = "\n".join(
        f"- **{v}:** {per_ad[CLEAN_ADS[0].id][v]['sample_question']}" for v in VERBOSITY
    )
    md += f"\n\nFirst question for `{CLEAN_ADS[0].id}`, per setting:\n\n{samples}\n"
    md += f"\nWhole experiment: {usage['calls']} calls, ${usage['cost_usd']:.4f}."
    return md, {"per_ad": per_ad, "usage": usage}


# --------------------------------------------------------------------------- main
EXPERIMENTS = {"max-tokens": max_tokens, "verbosity": verbosity}


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="One-setting comparison experiments.")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    name = parser.parse_args().experiment

    started = time.perf_counter()
    md, data = await EXPERIMENTS[name]()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    header = (f"# Experiment: {name}\n\n- **Run:** {stamp}\n- **Model:** `{SETTINGS.model}` "
              f"(reasoning effort: {SETTINGS.reasoning_effort})\n- **Prompt version:** "
              f"`{PROMPT_VERSION}`\n- **Ads:** {', '.join(a.id for a in CLEAN_ADS)}\n"
              f"- **Wall time:** {time.perf_counter() - started:.0f}s\n\n")

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"experiment-{name}-{stamp}.md").write_text(header + md + "\n", encoding="utf-8")
    (ARTIFACTS / f"experiment-{name}-{stamp}.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(header + md)
    print(f"\nWritten to artifacts/experiment-{name}-{stamp}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
