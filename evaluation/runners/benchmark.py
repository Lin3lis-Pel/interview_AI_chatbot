"""Run the AI-performance evaluation and write a report.

    uv run python -m evaluation.runners.benchmark                          # configured strategy, all suites
    uv run python -m evaluation.runners.benchmark --strategies all         # the 5-way prompting matrix
    uv run python -m evaluation.runners.benchmark --strategies zero_shot,few_shot
    uv run python -m evaluation.runners.benchmark --suite ads              # stage 1-3 only
    uv run python -m evaluation.runners.benchmark --suite answers          # stage 5-6 only
    uv run python -m evaluation.runners.benchmark --suite security
    uv run python -m evaluation.runners.benchmark --repeats 3              # variance across runs
    uv run python -m evaluation.runners.benchmark --ad devops_senior
    uv run python -m evaluation.runners.benchmark --strategies all --judge  # add the independent judge

Each strategy gets its own LLM client, so cost, tokens and latency are attributed per arm.
Strategies run sequentially (clean latency, no rate-limit interference); everything inside a
strategy runs concurrently.

Reports land in artifacts/ as JSON and Markdown, including a metric x strategy matrix.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tabulate import tabulate

from interview_ai.config import SETTINGS, STRATEGY_BLURB
from interview_ai.domain import InterviewSession
from interview_ai.llm import LLM, LLMUnavailable
from interview_ai.pipeline import NotAJobAd, analyze_job, generate_questions, plan_interview
from interview_ai.prompts import PROMPT_VERSION, STRATEGIES
from interview_ai.pipeline import build_report
from interview_ai.domain import Evaluation, Question
from evaluation import metrics as M
from evaluation import stats as S
from evaluation.artifacts import ArtifactLog
from evaluation.fixtures.answers import ADVERSARIAL_ANSWERS, ANSWERS, EXPECTED_ACTIONS, EXPECTED_BANDS
from evaluation.fixtures.job_ads import ALL_ADS, CLEAN_ADS, NOT_A_JOB_AD, AdFixture
from evaluation.judges.gemma import JudgeUnavailable, build_judge, judge_available, rate_questions, score_answer

REPORTS = Path(__file__).resolve().parents[2] / "artifacts"
TIERS = ("strong", "partial", "vague", "wrong", "no_answer")


# --------------------------------------------------------------------------- helpers
def _fake_question(fx) -> Question:
    return Question(topic=fx.topic, type="conceptual", text=fx.question,
                    targets=["RAG"], what_good_looks_like=fx.rubric)


def _mini_session(analysis, plan, question: Question, strategy: str, difficulty="medium") -> InterviewSession:
    return InterviewSession(analysis=analysis, plan=plan, questions=[question],
                            difficulty=difficulty, strategy=strategy)


# --------------------------------------------------------------------------- suite: ads
async def eval_one_ad(
    llm: LLM, ad: AdFixture, strategy: str, table: M.MetricTable, failures: list[str],
    judge: LLM | None = None, log: ArtifactLog | None = None, obs: S.Collected | None = None,
) -> None:
    def save(stage: str, payload, **meta):
        if log:
            log.write(strategy=strategy, stage=stage, fixture=ad.id, payload=payload, **meta)

    try:
        analysis = await analyze_job(llm, ad.text, strategy)
        table.add("schema_validity", 1.0)
        save("analysis", analysis)
    except (LLMUnavailable, NotAJobAd) as exc:
        table.add("schema_validity", 0.0)
        failures.append(f"[{ad.id}] analysis failed: {exc}")
        return

    if ad.adversarial:
        table.add("security_resisted_ad_injection",
                  M.resisted_ad_injection(analysis, ad.role, ad.gold_critical))
        table.add("security_no_prompt_leak", M.no_prompt_leak(analysis.notes, *analysis.responsibilities))
    else:
        table.add("role_accuracy", M.role_correct(analysis, ad.role))
        table.add("seniority_exact", M.seniority_exact(analysis, ad.seniority))
        table.add("seniority_within_one", M.seniority_within_one(analysis, ad.seniority))
        table.add("requirement_recall", M.requirement_recall(analysis, ad.must_have))
        table.add("criticality_precision", M.criticality_precision(analysis, ad.gold_critical))
        table.add("emphasis_detected", M.emphasis_detected(analysis))
        table.add("analysis_exemplar_clean",
                  M.analysis_exemplar_leakage(analysis, ad.role, ad.text))
        if obs is not None:
            obs.gold_roles.append(ad.role)
            obs.pred_roles.append(analysis.role)
            obs.requirement_prf.append(S.requirement_retrieval(
                ad.must_have, [r.name for r in analysis.requirements] + analysis.technologies
            ))
    table.add("evidence_grounding", M.evidence_grounding(analysis, ad.text))

    if M.role_correct(analysis, ad.role) < 1 and not ad.adversarial:
        failures.append(f"[{ad.id}] role: expected {ad.role!r}, got {analysis.role!r}")
    if not ad.adversarial and M.analysis_exemplar_leakage(analysis, ad.role, ad.text) < 1.0:
        failures.append(f"[{ad.id}] analysis carries worked-example content (exemplar leakage)")

    # ---- stage 2 + 3, both difficulties in parallel ----
    try:
        plan_easy, plan_hard = await asyncio.gather(
            plan_interview(llm, analysis, "easy", strategy),
            plan_interview(llm, analysis, "hard", strategy),
        )
        table.add("schema_validity", 1.0)
    except LLMUnavailable as exc:
        table.add("schema_validity", 0.0)
        failures.append(f"[{ad.id}] planning failed: {exc}")
        return

    save("plan", plan_easy, difficulty="easy")
    save("plan", plan_hard, difficulty="hard")

    for plan in (plan_easy, plan_hard):
        table.add("plan_weights_valid", M.weights_valid(plan))
        table.add("plan_topic_count_ok", M.topic_count_in_range(plan))
        table.add("plan_behavioral_balanced", M.behavioral_topic_balanced(plan))
        table.add("plan_topic_grounding", M.topic_grounding(plan, analysis))
        table.add("syllabus_coverage", M.syllabus_coverage(plan, ad.expected_topic_keywords))
        table.add("plan_exemplar_clean", M.plan_exemplar_leakage(plan, ad.text))
        if M.plan_exemplar_leakage(plan, ad.text) < 1.0:
            failures.append(f"[{ad.id}] plan reuses worked-example subjects or weight shape")

    try:
        qs_easy, qs_hard = await asyncio.gather(
            generate_questions(llm, analysis, plan_easy, "easy", 8, strategy),
            generate_questions(llm, analysis, plan_hard, "hard", 8, strategy),
        )
        table.add("schema_validity", 1.0)
    except LLMUnavailable as exc:
        table.add("schema_validity", 0.0)
        failures.append(f"[{ad.id}] question generation failed: {exc}")
        return

    save("questions", qs_easy, difficulty="easy")
    save("questions", qs_hard, difficulty="hard")

    for qs in (qs_easy, qs_hard):
        table.add("question_grounding", M.question_grounding(qs, analysis))
        table.add("question_type_diversity", M.question_type_diversity(qs))
        table.add("question_rubric_present", M.rubric_present(qs))
        table.add("question_duplicate_rate", M.duplicate_rate(qs))
        table.add("question_exemplar_clean", M.question_exemplar_leakage(qs, ad.text))
    table.add("difficulty_separation", M.difficulty_separation(qs_easy, qs_hard))

    if M.duplicate_rate(qs_hard) > 0.15:
        failures.append(f"[{ad.id}] near-duplicate questions at hard difficulty")

    # ---- independent judge on question quality (regex cannot measure realism) ----
    if judge is not None and not ad.adversarial:
        ratings = await rate_questions(judge, analysis, qs_hard, "hard")
        if ratings:
            table.add("judged_question_quality", M.judged_question_quality(ratings))
            table.add("judged_difficulty_fit", M.judged_difficulty_fit(ratings))
            save("judge_question_ratings", ratings, difficulty="hard",
                 judge_model=SETTINGS.judge_model)


async def suite_ads(
    llm: LLM, ads: list[AdFixture], strategy: str, judge: LLM | None = None,
    log: ArtifactLog | None = None, obs: S.Collected | None = None,
) -> tuple[M.MetricTable, list[str]]:
    table, failures = M.MetricTable(), []
    await asyncio.gather(
        *(eval_one_ad(llm, ad, strategy, table, failures, judge, log, obs) for ad in ads)
    )

    # Negative control: a non-job-ad must be refused, not analysed into a fake interview.
    try:
        await analyze_job(llm, NOT_A_JOB_AD, strategy)
        table.add("rejects_non_job_ad", 0.0)
        failures.append("[negative control] a page of team notes was accepted as a job ad")
    except NotAJobAd:
        table.add("rejects_non_job_ad", 1.0)
    except LLMUnavailable as exc:
        failures.append(f"[negative control] call failed: {exc}")
    return table, failures


# --------------------------------------------------------------------------- suite: answers
async def suite_answers(
    llm: LLM, strategy: str, judge: LLM | None = None,
    log: ArtifactLog | None = None, obs: S.Collected | None = None,
) -> tuple[M.MetricTable, list[str]]:
    """Score every scripted answer against the same question and rubric."""
    table, failures = M.MetricTable(), []
    ad = CLEAN_ADS[0]
    analysis = await analyze_job(llm, ad.text, strategy)
    plan = await plan_interview(llm, analysis, "medium", strategy)

    def save(stage: str, payload, fixture: str, **meta):
        if log:
            log.write(strategy=strategy, stage=stage, fixture=fixture, payload=payload, **meta)

    async def score(fx):
        session = _mini_session(analysis, plan, _fake_question(fx), strategy)
        try:
            return fx, await session.submit_answer(llm, fx.answer)
        except LLMUnavailable as exc:
            failures.append(f"[{fx.id}] evaluation failed: {exc}")
            return fx, None

    results = await asyncio.gather(*(score(fx) for fx in ANSWERS))

    means: dict[str, list[float]] = {}
    coverages: dict[str, list[float]] = {}
    scored: list[tuple[object, Evaluation]] = []
    for fx, evaluation in results:
        if evaluation is None:
            table.add("schema_validity", 0.0)
            continue
        table.add("schema_validity", 1.0)
        scored.append((fx, evaluation))
        table.add("adaptive_action_policy", M.action_matches_policy(evaluation, EXPECTED_ACTIONS[fx.tier]))
        table.add("score_band_accuracy", M.score_in_band(evaluation, EXPECTED_BANDS[fx.tier]))
        table.add("follow_up_style", M.follow_up_is_conversational(evaluation))
        table.add("security_no_prompt_leak", M.no_prompt_leak(evaluation.follow_up))
        table.add("coverage_matches_verdict", M.coverage_matches_verdict(evaluation))
        means.setdefault(fx.tier, []).append(evaluation.scores.mean)
        coverages.setdefault(fx.tier, []).append(evaluation.information_coverage)
        # Self-contained on purpose: everything the offline judge needs (question, rubric,
        # answer, role) rides along, so `python -m evaluation.runners.judge_answers` can re-score this
        # record later without touching the paid API.
        save("evaluation", evaluation, fx.id, tier=fx.tier, gold_band=list(EXPECTED_BANDS[fx.tier]),
             question=fx.question, rubric=fx.rubric, answer=fx.answer, topic=fx.topic,
             role=analysis.role, seniority=analysis.seniority)
        if obs is not None:
            obs.tiers.append(fx.tier)
            obs.tier_scores.append(evaluation.scores.mean)

        if not M.action_matches_policy(evaluation, EXPECTED_ACTIONS[fx.tier]):
            failures.append(
                f"[{fx.id}] tier {fx.tier}: next_action {evaluation.next_action!r} "
                f"not in {sorted(EXPECTED_ACTIONS[fx.tier])}"
            )
        if not M.score_in_band(evaluation, EXPECTED_BANDS[fx.tier]):
            failures.append(
                f"[{fx.id}] tier {fx.tier}: mean score {evaluation.scores.mean} outside {EXPECTED_BANDS[fx.tier]}"
            )

    mean_by_tier = {t: round(statistics.mean(v), 2) for t, v in means.items()}
    table.add("score_discrimination", M.discrimination(mean_by_tier))
    table.rows["_mean_score_by_tier"] = [mean_by_tier.get(t, 0.0) for t in TIERS]

    # Coverage now decides whether the interview follows up and whether a requirement
    # counts as demonstrated, so it is measured for ordering just like the scores.
    coverage_by_tier = {t: round(statistics.mean(v), 2) for t, v in coverages.items()}
    table.add("coverage_discrimination", M.coverage_discrimination(coverage_by_tier))
    table.rows["_mean_coverage_by_tier"] = [coverage_by_tier.get(t, 0.0) for t in TIERS]

    # ---- cross-judge: an independent model re-scores the same answers ----
    if judge is not None and scored:
        async def cross(fx, primary):
            try:
                jr = await score_answer(
                    judge, role=analysis.role, seniority=analysis.seniority, topic=fx.topic,
                    question=fx.question, rubric=fx.rubric, answer=fx.answer,
                )
                return fx, primary, jr
            except LLMUnavailable as exc:
                failures.append(f"[{fx.id}] judge failed: {exc}")
                return fx, primary, None

        biases: list[float] = []
        for fx, primary, jr in await asyncio.gather(*(cross(f, e) for f, e in scored)):
            if jr is None:
                continue
            table.add("judge_score_agreement", M.judge_score_agreement(primary, jr))
            table.add("judge_verdict_agreement", M.judge_verdict_agreement(primary, jr))
            bias = M.judge_score_bias(primary, jr)
            table.add("judge_score_bias", bias)
            biases.append(bias)
            save("judge_answer_score", jr, fx.id, tier=fx.tier,
                 judge_model=SETTINGS.judge_model,
                 primary_mean=primary.scores.mean, primary_verdict=primary.verdict)
            if obs is not None:
                obs.primary_scores.append(primary.scores.mean)
                obs.judge_scores.append(jr.scores.mean)
                obs.primary_verdicts.append(primary.verdict)
                obs.judge_verdicts.append(jr.verdict)
            if M.judge_score_agreement(primary, jr) == 0.0:
                failures.append(
                    f"[{fx.id}] judge disagreement: primary {primary.scores.mean} "
                    f"({primary.verdict}) vs judge {jr.scores.mean} ({jr.verdict})"
                )
        if biases and statistics.mean(biases) > 0.5:
            failures.append(
                f"[cross-judge] primary model scored its own work {statistics.mean(biases):+.2f} "
                "higher than the independent judge on average -- possible self-preference bias"
            )

    # ---- final report end to end ----
    session = _mini_session(analysis, plan, _fake_question(ANSWERS[2]), strategy)  # vague answer
    try:
        await session.submit_answer(llm, ANSWERS[2].answer)
        report = await build_report(llm, session, strategy)
        table.add("schema_validity", 1.0)
        table.add("report_flags_unmet_requirements", float(bool(report.unmet_requirements)))
        table.add("report_gives_study_plan", float(bool(report.study_plan)))
        table.add("report_score_matches_answers", float(report.overall_score <= 65))
        table.add("security_no_prompt_leak",
                  M.no_prompt_leak(report.recommendation_reason, *report.study_plan))
        save("final_report", report, "vague_only_interview")
        if report.overall_score > 65:
            failures.append(f"[report] a vague-only interview scored {report.overall_score}/100")
    except LLMUnavailable as exc:
        table.add("schema_validity", 0.0)
        failures.append(f"[report] failed: {exc}")

    return table, failures


# --------------------------------------------------------------------------- suite: security
async def suite_security(llm: LLM, strategy: str) -> tuple[M.MetricTable, list[str]]:
    table, failures = M.MetricTable(), []
    ad = CLEAN_ADS[0]
    analysis = await analyze_job(llm, ad.text, strategy)
    plan = await plan_interview(llm, analysis, "medium", strategy)

    async def probe(fx):
        session = _mini_session(analysis, plan, _fake_question(fx), strategy)
        try:
            return fx, await session.submit_answer(llm, fx.answer)
        except LLMUnavailable as exc:
            failures.append(f"[{fx.id}] failed: {exc}")
            return fx, None

    for fx, evaluation in await asyncio.gather(*(probe(f) for f in ADVERSARIAL_ANSWERS)):
        if evaluation is None:
            table.add("schema_validity", 0.0)
            continue
        table.add("schema_validity", 1.0)
        leak = M.no_prompt_leak(evaluation.follow_up, *evaluation.missing, *evaluation.covered)
        resisted = M.resisted_score_injection(evaluation)
        table.add("security_no_prompt_leak", leak)
        table.add("security_resisted_score_injection", resisted)
        if not leak:
            failures.append(f"[{fx.id}] SYSTEM PROMPT LEAK in the follow-up")
        if not resisted:
            failures.append(
                f"[{fx.id}] injection moved the score to {evaluation.scores.mean} "
                f"(verdict {evaluation.verdict})"
            )
    return table, failures


# --------------------------------------------------------------------------- one strategy arm
async def run_strategy(
    strategy: str, suites: list[str], ads: list[AdFixture], repeats: int, judge: LLM | None,
    log: ArtifactLog | None = None,
) -> dict:
    llm = LLM()
    table, failures = M.MetricTable(), []
    obs = S.Collected()
    started = time.perf_counter()

    for run_no in range(repeats):
        for suite in suites:
            print(f"  [{strategy}] run {run_no + 1}/{repeats}: {suite}...", flush=True)
            try:
                if suite == "ads":
                    t, f = await suite_ads(llm, ads, strategy, judge, log, obs)
                elif suite == "answers":
                    t, f = await suite_answers(llm, strategy, judge, log, obs)
                else:
                    t, f = await suite_security(llm, strategy)
                table.merge(t)
                failures.extend(f)
            except Exception as exc:  # noqa: BLE001 - one broken suite must not lose the rest
                failures.append(f"[suite {suite}] crashed: {type(exc).__name__}: {exc}")

    usage = llm.usage.summary()
    await llm.aclose()

    tiers = table.rows.pop("_mean_score_by_tier", None)
    cov_tiers = table.rows.pop("_mean_coverage_by_tier", None)
    summary = table.summary()
    summary.pop("_mean_score_by_tier", None)
    summary.pop("_mean_coverage_by_tier", None)
    thresholded = {k: v for k, v in summary.items() if k in M.THRESHOLDS}
    n_passed = sum(M.verdict(k, v["mean"]) for k, v in thresholded.items())

    return {
        "strategy": strategy,
        "wall_time_s": round(time.perf_counter() - started, 1),
        "usage": usage,
        "metrics": summary,
        "mean_score_by_tier": dict(zip(TIERS, tiers)) if tiers else None,
        "mean_coverage_by_tier": dict(zip(TIERS, cov_tiers)) if cov_tiers else None,
        "stats": obs.summary(),
        "n_thresholded": len(thresholded),
        "n_passed": n_passed,
        "passed": n_passed == len(thresholded),
        "failures": failures,
    }


# --------------------------------------------------------------------------- scoring the arms
def composite(arm: dict, metric_names: list[str]) -> float:
    """Mean of thresholded metrics, orientation-corrected, over metrics all arms share."""
    vals = []
    for name in metric_names:
        stats = arm["metrics"].get(name)
        if not stats:
            continue
        v = stats["mean"]
        vals.append(1.0 - v if name in M.LOWER_IS_BETTER else v)
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def shared_metric_names(arms: list[dict]) -> list[str]:
    """Only compare on metrics every arm actually produced -- otherwise the matrix lies."""
    sets = [set(a["metrics"]) & set(M.THRESHOLDS) for a in arms]
    return sorted(set.intersection(*sets)) if sets else []


def _best(name: str, values: dict[str, float]) -> str:
    if not values:
        return "-"
    return (min if name in M.LOWER_IS_BETTER else max)(values, key=values.get)


# --------------------------------------------------------------------------- reporting
def _table(headers: list[str], rows: list[list]) -> list[str]:
    """One GitHub-flavoured markdown table. First column left-aligned, the rest right.

    Cells arrive pre-formatted as strings so tabulate never re-rounds a number.
    """
    if not rows:
        return ["_(no data)_", ""]
    return [
        # disable_numparse: cells are already formatted to the precision this report wants.
        # Without it tabulate re-parses them as floats and re-rounds -- p=0.0029 became 0.003.
        tabulate(rows, headers=headers, tablefmt="github", disable_numparse=True,
                 colalign=["left"] + ["right"] * (len(headers) - 1)),
        "",
    ]


def _tier_table(arms: list[dict], key: str, metric: str) -> list[str]:
    """Mean value per answer tier, one row per strategy, plus the ordering metric."""
    return _table(
        ["Strategy", *TIERS, "ordered"],
        [
            [arm["strategy"]]
            + [f"{arm[key].get(t, 0.0):.2f}" for t in TIERS]
            + [f"{arm['metrics'].get(metric, {}).get('mean', 0.0):.2f}"]
            for arm in arms if arm.get(key)
        ],
    )


def _stat_table(arms: list[dict], stat: str, headers: list[str],
                fields: list[tuple[str, str]]) -> list[str]:
    """One row per arm from `arm["stats"][stat]`.

    `fields` is (key, format-spec) per column -- per-column because the precision differs:
    a p-value needs .4f, a signed bias needs +.3f, and a count is an int.
    """
    rows = []
    for arm in arms:
        st = arm["stats"].get(stat)
        if not st:
            continue
        rows.append([arm["strategy"]]
                    + [format(st.get(key, float("nan")), spec) for key, spec in fields])
    return _table(headers, rows)


def render_markdown(payload: dict) -> str:
    arms: list[dict] = payload["arms"]
    strategies = [a["strategy"] for a in arms]
    multi = len(arms) > 1

    lines = [
        "# interview_AI - AI performance report",
        "",
        f"- **Run:** {payload['timestamp']}",
        f"- **Model:** `{payload['model']}` (reasoning effort: {payload['reasoning_effort']})",
        f"- **Prompt version:** `{payload['prompt_version']}`",
        f"- **Strategies:** {', '.join(strategies)}",
        f"- **Suites:** {', '.join(payload['suites'])} - **repeats:** {payload['repeats']}",
        f"- **Independent judge:** "
        + (f"`{payload['judge']}` (strategy: {payload.get('judge_strategy', '-')})"
           if payload["judge"] else "not enabled"),
        f"- **Total wall time:** {payload['wall_time_s']:.1f}s",
        f"- **Raw outputs:** {payload.get('artifact_records', 0)} records in "
        f"`{Path(payload['artifacts']).name}`" if payload.get("artifacts") else
        "- **Raw outputs:** not persisted",
        "",
    ]

    # ---- headline ----
    for arm in arms:
        u = arm["usage"]
        lines.append(
            f"**{arm['strategy']}** - {'PASS' if arm['passed'] else 'FAIL'} "
            f"({arm['n_passed']}/{arm['n_thresholded']} thresholds met) - "
            f"${u['cost_usd']:.4f}, {u['total_tokens']:,} tokens, "
            f"p50 {u['p50_latency_s']:.2f}s"
        )
    lines.append("")

    # ---- the matrix ----
    names = shared_metric_names(arms) if multi else sorted(arms[0]["metrics"])
    all_names = sorted({n for a in arms for n in a["metrics"] if not n.startswith("_")})

    rows = []
    for name in all_names:
        values = {a["strategy"]: a["metrics"][name]["mean"] for a in arms if name in a["metrics"]}
        threshold = M.THRESHOLDS.get(name)
        target = ("diagnostic" if threshold is None else
                  f"{'<=' if name in M.LOWER_IS_BETTER else '>='} {threshold:.2f}")
        best = _best(name, values) if multi and threshold is not None else None

        cells = [name]
        for s in strategies:
            if s not in values:
                cells.append("-")
                continue
            v = values[s]
            mark = "**" if best == s else ""
            flag = "" if threshold is None or M.verdict(name, v) else " ⚠"
            cells.append(f"{mark}{v:.3f}{mark}{flag}")
        if multi:
            cells.append(best or "-")
        cells.append(target)
        rows.append(cells)

    lines += ["## Metric x strategy matrix" if multi else "## Metrics", ""]
    lines += _table(
        ["Metric", *strategies] + (["Best"] if multi else []) + ["Threshold"], rows
    )
    lines += ["⚠ marks a value the wrong side of its threshold.", ""]

    # ---- composite ranking ----
    if multi and names:
        ranked = sorted(arms, key=lambda a: composite(a, names), reverse=True)
        lines += [
            "## Which prompting strategy wins",
            "",
            f"Composite = mean of the {len(names)} thresholded metrics every arm produced, "
            "orientation-corrected. Cost is reported alongside but not folded in.",
            "",
        ]
        lines += _table(
            ["Rank", "Strategy", "Composite", "Thresholds met", "Cost", "Tokens", "p50 latency"],
            [[str(i), arm["strategy"], f"{composite(arm, names):.4f}",
              f"{arm['n_passed']}/{arm['n_thresholded']}",
              f"${arm['usage']['cost_usd']:.4f}", f"{arm['usage']['total_tokens']:,}",
              f"{arm['usage']['p50_latency_s']:.2f}s"]
             for i, arm in enumerate(ranked, 1)],
        )
        winner = ranked[0]
        cheapest = min(arms, key=lambda a: a["usage"]["cost_usd"])
        lines += [
            f"**Best quality:** `{winner['strategy']}` - {STRATEGY_BLURB.get(winner['strategy'], '')}",
            "",
            f"**Cheapest:** `{cheapest['strategy']}` at ${cheapest['usage']['cost_usd']:.4f} "
            f"({composite(cheapest, names):.4f} composite).",
            "",
        ]
        # Cost per composite point, so a marginal quality gain at 3x cost is visible.
        lines += _table(
            ["Strategy", "Composite", "$ per run", "$ per composite point"],
            [[arm["strategy"], f"{composite(arm, names):.4f}",
              f"${arm['usage']['cost_usd']:.4f}",
              f"${arm['usage']['cost_usd'] / composite(arm, names):.4f}"
              if composite(arm, names) else "-"]
             for arm in ranked],
        )

    # ---- scorer discrimination per strategy ----
    if any(a.get("mean_score_by_tier") for a in arms):
        lines += [
            "## Scorer discrimination",
            "",
            "Mean 1-5 score given to each scripted answer tier. Values must decrease left to "
            "right; if they do not, the scorer is not separating good answers from bad ones.",
            "",
        ]
        lines += _tier_table(arms, "mean_score_by_tier", "score_discrimination")

    # ---- coverage per tier (the signal that now drives the interview loop) ----
    if any(a.get("mean_coverage_by_tier") for a in arms):
        lines += [
            "## Coverage discrimination",
            "",
            "Mean `information_coverage` (0-1) given to each answer tier. This is the number "
            "the interview loop runs on: reaching the difficulty threshold ends the question, "
            "falling short buys a follow-up, and a question that never gets there is reported "
            "as an unmet requirement. It must fall left to right.",
            "",
        ]
        lines += _tier_table(arms, "mean_coverage_by_tier", "coverage_discrimination")

    # ---- cross-judge ----
    if payload["judge"]:
        lines += [
            "## Independent judge",
            "",
            f"Second opinion from `{payload['judge']}`, which never sees the primary score. "
            "`judge_score_bias` is signed: positive means the primary model rated its own "
            "generations higher than the judge did, which is the self-preference signature.",
            "",
        ]
        lines += _table(
            ["Strategy", "score agreement", "verdict agreement", "mean bias", "question quality"],
            [[arm["strategy"]]
             + [f"{arm['metrics'].get(k, {}).get('mean', float('nan')):{f}}"
                for k, f in (("judge_score_agreement", ".3f"),
                             ("judge_verdict_agreement", ".3f"),
                             ("judge_score_bias", "+.3f"),
                             ("judged_question_quality", ".3f"))]
             for arm in arms],
        )

    # ---- statistics (scikit-learn / scipy) ----
    if any(a.get("stats") for a in arms):
        lines += [
            "## Statistics",
            "",
            "Computed with scikit-learn and scipy. These answer *how well* rather than "
            "*did it hold*, which is what the thresholded metrics above cover.",
            "",
        ]

        if any(a["stats"].get("role_classification") for a in arms):
            lines += ["### Role classification", ""]
            lines += _stat_table(arms, "role_classification",
                                 ["Strategy", "accuracy", "macro F1"],
                                 [("accuracy", ".3f"), ("macro_f1", ".3f")])
            # One confusion matrix, from the first arm that has one: a systematic mix-up
            # (every ML Engineer ad read as Data Scientist) is invisible in an accuracy number.
            for arm in arms:
                rc = arm["stats"].get("role_classification")
                if rc and rc.get("confusion_matrix"):
                    lines += [f"Confusion matrix (`{arm['strategy']}`), rows = gold:", ""]
                    lines += S.render_confusion(rc)
                    lines.append("")
                    break

        if any(a["stats"].get("requirement_retrieval") for a in arms):
            lines += [
                "### Requirement extraction",
                "",
                "Recall alone rewards a model that lists every noun in the ad; precision is "
                "what stops it.",
                "",
            ]
            lines += _stat_table(arms, "requirement_retrieval",
                                 ["Strategy", "precision", "recall", "F1", "ads"],
                                 [("precision", ".3f"), ("recall", ".3f"),
                                  ("f1", ".3f"), ("n", "d")])

        if any(a["stats"].get("tier_rank_correlation") for a in arms):
            lines += [
                "### Scorer rank correlation",
                "",
                "Spearman rho between the gold answer tier and the score given. A stricter "
                "reading than the adjacent-pair fraction: near 1.0 means the scorer tracks "
                "answer quality, near 0 means it is not measuring the answer at all. "
                "`nan` means one side was constant (e.g. everything scored the same).",
                "",
            ]
            lines += _stat_table(arms, "tier_rank_correlation", ["Strategy", "rho", "p", "n"],
                                 [("spearman_rho", ".3f"), ("spearman_p", ".4f"),
                                  ("n", "d")])

        if any(a["stats"].get("judge_agreement") for a in arms):
            lines += [
                "### Inter-rater reliability vs the independent judge",
                "",
                "Cohen's kappa (quadratic-weighted, because the verdict scale is ordinal) "
                "corrects for the agreement two scorers reach by chance. `mean_bias` is "
                "signed: positive means the primary model rated its own generations above "
                "the second opinion.",
                "",
            ]
            lines += _stat_table(
                arms, "judge_agreement",
                ["Strategy", "kappa", "rho", "MAE", "mean bias", "exact match", "n"],
                [("cohen_kappa", ".3f"), ("spearman_rho", ".3f"), ("mae", ".3f"),
                 ("mean_bias", "+.3f"), ("exact_verdict_match", ".3f"), ("n", "d")],
            )

    # ---- failures ----
    lines += ["## Failures", ""]
    for arm in arms:
        lines.append(f"### {arm['strategy']} ({len(arm['failures'])})")
        lines += [f"- {f}" for f in arm["failures"][:40]] or ["- none"]
        if len(arm["failures"]) > 40:
            lines.append(f"- ... and {len(arm['failures']) - 40} more")
        lines.append("")

    lines += [
        "## How to read this",
        "",
        "- **evidence_grounding** is the anti-hallucination metric: every extracted requirement "
        "must be quotable from the ad.",
        "- **question_grounding / plan_topic_grounding** check the interview is built from the "
        "ad, not from the model's idea of the role.",
        "- **score_discrimination** is the single most important number: if it is not 1.0, the "
        "final report is noise.",
        "- **\\*_exemplar_clean** is the cost side of few-shot prompting: 1.0 means the output "
        "carries no content copied from the worked examples. Expect few_shot to score worst "
        "here and best on schema_validity -- that trade-off is the reason both are measured.",
        "- **security_\\*** metrics must be 1.0. Anything less is a live prompt-injection hole.",
        "- **judge_score_bias** consistently positive across tiers means the primary model is "
        "marking its own homework generously; the gold bands in evaluation/fixtures/answers/, not "
        "the judge, remain the ground truth.",
        "- Bump the version in `interview_ai/prompts/VERSION.txt` when tuning, then diff two reports.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
async def main() -> int:
    # The Windows console defaults to cp1252, which cannot print the report's unicode
    # (e.g. the warning sign). The files are always written UTF-8; this keeps the final
    # console echo from crashing an otherwise-finished run.
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="interview_AI evaluation harness")
    parser.add_argument("--suite", choices=["all", "ads", "answers", "security"], default="all")
    parser.add_argument("--strategies", default=SETTINGS.strategy,
                        help="comma list, or 'all' for the full prompting matrix")
    parser.add_argument("--repeats", type=int, default=1, help="repeat suites to measure variance")
    parser.add_argument("--ad", help="run a single ad fixture by id")
    parser.add_argument("--judge", action="store_true",
                        help="enable the independent judge (needs [judge] model in config/settings.toml)")
    parser.add_argument("--no-artifacts", action="store_true",
                        help="do not persist raw model outputs to JSONL")
    args = parser.parse_args()

    if not SETTINGS.configured:
        print("OPENROUTER_API_KEY is not set. Paste your key into .env first.")
        return 2

    if args.strategies.strip().lower() == "all":
        strategies = list(STRATEGIES)
    else:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        print(f"Unknown strategies {unknown}; expected any of {list(STRATEGIES)}.")
        return 2

    ads = [a for a in ALL_ADS if a.id == args.ad] if args.ad else ALL_ADS
    if args.ad and not ads:
        print(f"No fixture with id {args.ad!r}.")
        return 2
    suites = ["ads", "answers", "security"] if args.suite == "all" else [args.suite]

    # ---- optional independent judge ----
    judge: LLM | None = None
    judge_name = ""
    if args.judge:
        try:
            judge = build_judge()
            if await judge_available(judge):
                judge_name = SETTINGS.judge_model
                print(f"Independent judge: {judge_name} at {SETTINGS.judge_base_url}")
            else:
                print(f"Judge {SETTINGS.judge_model} did not respond at "
                      f"{SETTINGS.judge_base_url}; continuing without it. Is `ollama serve` running?")
                await judge.aclose()
                judge = None
        except (JudgeUnavailable, LLMUnavailable) as exc:
            print(f"Judge unavailable, continuing without it: {exc}")
            judge = None

    # Artifacts are opened BEFORE any call, and appended as results arrive, so a crash keeps
    # everything produced up to that point. The aggregate report is written only at the end.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = ArtifactLog(REPORTS / f"artifacts-{stamp}.jsonl", run_id=stamp,
                      enabled=not args.no_artifacts)
    if log.enabled:
        print(f"Raw model outputs -> {log.path}")

    started = time.perf_counter()
    arms = [await run_strategy(s, suites, ads, args.repeats, judge, log) for s in strategies]
    if judge is not None:
        await judge.aclose()

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": SETTINGS.model,
        "reasoning_effort": SETTINGS.reasoning_effort,
        "prompt_version": PROMPT_VERSION,
        "suites": suites,
        "repeats": args.repeats,
        "judge": judge_name,
        "judge_strategy": SETTINGS.judge_strategy if judge_name else "",
        "max_output_tokens": {s: SETTINGS.max_tokens_for(s)
                              for s in ("analyzer", "planner", "questioner", "evaluator", "reporter")},
        "wall_time_s": time.perf_counter() - started,
        "artifacts": str(log.path) if log.enabled else "",
        "artifact_records": log.count,
        "arms": arms,
        "passed": all(a["passed"] for a in arms),
    }

    # Flat metric x strategy view inside the same JSON, so comparing two runs is one load()
    # and a dict diff -- no second file, no spreadsheet.
    try:
        payload["matrix"] = S.matrix_frame(arms).to_dict()
        payload["usage_table"] = S.usage_frame(arms).to_dict(orient="index")
    except Exception as exc:  # noqa: BLE001 - the report itself is the deliverable
        payload["matrix_error"] = str(exc)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"eval-{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = render_markdown(payload)
    (REPORTS / f"eval-{stamp}.md").write_text(markdown, encoding="utf-8")
    (REPORTS / "latest.md").write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"\nWritten to {REPORTS / f'eval-{stamp}.md'}")
    if log.enabled:
        print(f"{log.count} raw model outputs in {log.path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
