"""interview_AI -- Streamlit frontend.

Three screens driven by one `stage` value in session_state:
    setup -> interview -> report

All model work happens in interview_ai/pipeline; this file only renders and dispatches.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

import streamlit as st

from interview_ai.config import ALLOWED_MODELS, DIFFICULTY_RUBRIC, SETTINGS, STRATEGY_BLURB
from interview_ai.domain import InterviewSession
from interview_ai.llm import LLM, LLMUnavailable
from interview_ai.pipeline import (
    RECOMMENDATION_LABEL,
    NotAJobAd,
    build_report,
    prepare_interview,
    report_markdown,
)
from interview_ai.prompts import PERSONAS, VERBOSITY
from interview_ai.utils.jobsource import FetchError, fetch_job_ad
from interview_ai.utils.security import RateLimiter, scan_input

st.set_page_config(page_title="interview_AI", page_icon="🎯", layout="centered")

PERSONA_BLURB = {
    "neutral": "Professional and even. Gives away nothing about how it is going.",
    "friendly": "Warm and encouraging, but still asks the hard question.",
    "strict": "Terse and demanding. Presses for numbers and decisions, never for comfort.",
}

RISK_UI = {
    "low": ("✅", "Input scanned - clean."),
    "medium": ("⚠️", "Input scanned - suspicious phrasing found. It is fenced as data, not instructions."),
    "high": ("🛑", "Input scanned - looks like a prompt-injection attempt. Fenced as data; review before continuing."),
}


# --------------------------------------------------------------------------- plumbing
def run(coro):
    """Streamlit reruns are synchronous; each async batch gets its own loop."""
    return asyncio.run(coro)


def get_llm() -> LLM:
    """One LLM per action, built from whatever the sidebar currently says."""
    if "limiter" not in st.session_state:
        st.session_state.limiter = RateLimiter()
    settings = st.session_state.get("run_settings", SETTINGS)
    return LLM(settings=settings, limiter=st.session_state.limiter)


def init_state() -> None:
    st.session_state.setdefault("stage", "setup")
    st.session_state.setdefault("session", None)
    st.session_state.setdefault("report", None)
    st.session_state.setdefault("job_ad", "")
    st.session_state.setdefault("usage", None)
    st.session_state.setdefault("answer_nonce", 0)


def reset() -> None:
    for key in ("stage", "session", "report", "job_ad", "usage", "limiter"):
        st.session_state.pop(key, None)
    init_state()


# --------------------------------------------------------------------------- sidebar
# Every model setting is tunable here. The controls build a per-session `Settings` via
# `dataclasses.replace`, which the session's LLM then uses -- so nothing mutates global
# state and config/settings.toml stays the defaults, not the live values.
def sidebar() -> tuple[str, str, str]:
    locked = st.session_state.stage != "setup"

    with st.sidebar:
        st.subheader("Interview")
        difficulty = st.select_slider(
            "Difficulty", options=["easy", "medium", "hard"], value="medium", disabled=locked,
        )
        st.caption(DIFFICULTY_RUBRIC[difficulty])

        n = st.slider("Questions", 3, 16, SETTINGS.default_questions, disabled=locked)
        st.session_state.n_questions = n

        personas = list(PERSONAS)
        persona = st.radio(
            "Interviewer manner", personas,
            index=personas.index(SETTINGS.persona) if SETTINGS.persona in personas else 0,
            horizontal=True, disabled=locked,
        )
        st.caption(PERSONA_BLURB[persona])

        lengths = list(VERBOSITY)
        verbosity = st.radio(
            "Response length", lengths,
            index=lengths.index(SETTINGS.verbosity) if SETTINGS.verbosity in lengths else 0,
            horizontal=True, disabled=locked,
            help="Length of the questions and the final report. Scoring is unaffected.",
        )

        st.divider()
        st.caption("**Follow-ups**")
        followups = st.slider(
            "Max follow-ups per question", 0, 3, SETTINGS.max_followups_per_question,
            disabled=locked,
            help="0 turns follow-ups off entirely: a straight question list.",
        )
        default_threshold = SETTINGS.coverage_threshold_for(difficulty)
        threshold = st.slider(
            "Answer completeness needed to move on", 0.0, 1.0, float(default_threshold), 0.05,
            disabled=locked,
            help="An answer scoring at or above this covers enough of the question's rubric, "
                 "and the interview advances instead of following up.",
        )
        st.caption(
            f"At {threshold:.0%} an answer must cover most of what the question asks before "
            "moving on. Lower it for a faster interview, raise it for a harder one."
        )

        st.divider()
        st.caption("**Prompting strategy**")
        options = list(STRATEGY_BLURB)
        strategy = st.selectbox(
            "Prompting strategy", options,
            index=options.index(SETTINGS.strategy) if SETTINGS.strategy in options else 0,
            disabled=locked, label_visibility="collapsed",
        )
        st.caption(STRATEGY_BLURB[strategy])

        with st.expander("Model settings"):
            models = list(ALLOWED_MODELS)
            model = st.selectbox(
                "Model", models,
                index=models.index(SETTINGS.model) if SETTINGS.model in models else 0,
                disabled=locked, help="Only the Sprint 1 allow-listed OpenRouter models.",
            )
            st.caption(ALLOWED_MODELS[model])
            efforts = ["minimal", "low", "medium", "high"]
            effort = st.select_slider(
                "Reasoning effort", options=efforts,
                value=SETTINGS.reasoning_effort if SETTINGS.reasoning_effort in efforts else "low",
                disabled=locked,
            )
            uses_reasoning = "gpt-5" in model.lower()
            temperature = st.slider(
                "Temperature", 0.0, 2.0, float(SETTINGS.temperature), 0.1,
                disabled=locked or uses_reasoning,
            )
            if uses_reasoning:
                st.caption(
                    "GPT-5 uses reasoning effort. Temperature is disabled and is not sent "
                    "to the model."
                )
            st.caption("**Max output tokens per stage**")
            max_tokens = {
                stage: st.number_input(
                    stage.capitalize(), 500, 20_000, SETTINGS.max_tokens_for(stage), step=500,
                    disabled=locked, key=f"maxtok_{stage}",
                )
                for stage in ("analyzer", "planner", "questioner", "evaluator", "reporter")
            }
            st.caption(
                "On gpt-5 the reasoning tokens are billed against these, so a tight cap "
                "truncates the JSON and costs more than it saves."
            )

        # One Settings object for this session, handed to the LLM by get_llm().
        st.session_state.run_settings = replace(
            SETTINGS,
            model=model,
            reasoning_effort=effort,
            temperature=temperature,
            persona=persona,
            verbosity=verbosity,
            max_followups_per_question=followups,
            max_tokens_overrides=max_tokens,
            coverage_override=threshold,
        )

        st.divider()
        if not SETTINGS.configured:
            st.error("No API key. Paste it into `.env` as `OPENROUTER_API_KEY=...`, then restart.")
        if limiter := st.session_state.get("limiter"):
            st.caption(f"Call budget: {limiter.used}/{limiter.max_calls}")
        if usage := st.session_state.get("usage"):
            st.caption(
                f"{usage['calls']} calls - {usage['total_tokens']:,} tokens - "
                f"${usage['cost_usd']:.4f} - p50 {usage['p50_latency_s']:.1f}s"
            )

        st.divider()
        if st.button("Start over", use_container_width=True):
            reset()
            st.rerun()
    return difficulty, strategy, persona, verbosity


# --------------------------------------------------------------------------- setup
def screen_setup(difficulty: str, strategy: str, persona: str, verbosity: str) -> None:
    st.title("🎯 interview_AI")
    st.write("Drop a job ad. Get an interview built from it, then a report on how you did.")

    tab_paste, tab_url = st.tabs(["Paste the ad", "LinkedIn / job URL"])
    job_ad = ""

    with tab_paste:
        job_ad = st.text_area(
            "Job description", height=280, value=st.session_state.job_ad,
            placeholder="Paste the whole posting - title, responsibilities and requirements.",
        )

    with tab_url:
        url = st.text_input("Job ad URL", placeholder="https://www.linkedin.com/jobs/view/...")
        st.caption(
            "LinkedIn blocks most automated reads. If the fetch fails, open the posting and paste it."
        )
        if st.button("Fetch", disabled=not url):
            with st.spinner("Fetching..."):
                try:
                    st.session_state.job_ad = run(fetch_job_ad(url))
                    st.success(f"Fetched {len(st.session_state.job_ad):,} characters. "
                               "Check it on the paste tab, then start.")
                    st.rerun()
                except FetchError as exc:
                    st.error(str(exc))

    if job_ad != st.session_state.job_ad:
        st.session_state.job_ad = job_ad

    text = st.session_state.job_ad
    if not text:
        return

    verdict = scan_input(text, max_chars=SETTINGS.max_job_ad_chars, kind="job ad")
    icon, message = RISK_UI[verdict.risk]
    (st.success if verdict.risk == "low" else st.warning)(f"{icon} {message}")
    if verdict.reasons:
        st.caption("Detected: " + ", ".join(verdict.reasons))
    for note in verdict.notes:
        st.caption(f"- {note}")

    if st.button("Start interview", type="primary", disabled=not SETTINGS.configured):
        with st.spinner("Reading the ad, planning the interview, writing questions..."):
            try:
                llm = get_llm()
                analysis, plan, questions = run(
                    prepare_interview(llm, verdict.text, difficulty,
                                      st.session_state.n_questions, strategy, persona,
                                      verbosity)
                )
            except (NotAJobAd, LLMUnavailable) as exc:
                st.error(str(exc))
                return
            except PermissionError as exc:
                st.error(str(exc))
                return

        st.session_state.session = InterviewSession(
            analysis=analysis, plan=plan, questions=questions,
            difficulty=difficulty, strategy=strategy, persona=persona, verbosity=verbosity,
            settings=st.session_state.get("run_settings", SETTINGS),
        )
        st.session_state.usage = llm.usage.summary()
        st.session_state.stage = "interview"
        st.rerun()


# --------------------------------------------------------------------------- interview
def render_plan(session: InterviewSession, expanded: bool = False) -> None:
    a = session.analysis
    with st.expander("What the ad asked for", expanded=expanded):
        st.markdown(f"**{a.title or 'Untitled'}** {'@ ' + a.company if a.company else ''}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Role", a.role)
        c2.metric("Seniority", a.seniority)
        c3.metric("Difficulty", session.difficulty)
        if a.emphasis:
            st.markdown("**The company keeps coming back to:** " + ", ".join(a.emphasis))
        for level, label in (("critical", "Critical"), ("important", "Important"), ("bonus", "Bonus")):
            names = [r.name for r in a.requirements if r.criticality == level]
            if names:
                st.markdown(f"**{label}:** {', '.join(names)}")

    with st.expander("Interview plan", expanded=expanded):
        for t in session.plan.topics:
            st.markdown(f"**{t.name}** - {t.weight}%")
            st.progress(t.weight / 100)
            st.caption(t.rationale)


def screen_interview() -> None:
    session: InterviewSession = st.session_state.session
    done, total = session.progress
    st.progress((done - 1) / total, text=f"Question {done} of {total}")
    render_plan(session)

    prompt = session.current_prompt()
    if prompt is None:
        st.session_state.stage = "report"
        st.rerun()

    q = session.current_question
    # Say plainly when this is a follow-up and that it is the only one: the progress bar
    # counts base questions, so without this the interview looks stuck on question 1.
    if session.pending_followup:
        label = f"Subject: {q.topic}  -  follow-up {session.followups_used} of {session.max_followups}"
    else:
        label = f"Subject: {q.topic}  -  {q.type}"
    st.caption(label)
    st.markdown(f"### {prompt}")

    with st.form(key=f"answer_{st.session_state.answer_nonce}", clear_on_submit=True):
        answer = st.text_area("Your answer", height=200, label_visibility="collapsed",
                              placeholder="Answer as you would out loud.")
        c1, c2, c3 = st.columns([2, 1, 1])
        submitted = c1.form_submit_button("Submit", type="primary", use_container_width=True)
        skipped = c2.form_submit_button("Skip", use_container_width=True)
        ended = c3.form_submit_button("End", use_container_width=True)

    if skipped:
        session.skip()
        st.session_state.answer_nonce += 1
        st.rerun()

    if ended:
        session.end_early()
        st.session_state.stage = "report"
        st.rerun()

    if submitted:
        verdict = scan_input(answer, max_chars=SETTINGS.max_answer_chars, kind="answer")
        if verdict.blocked:
            st.warning("Write something first, or press Skip.")
            return
        if verdict.risk != "low":
            st.info(
                "Heads up: that answer contains instruction-like text. It is scored as an "
                "answer, not obeyed as a command."
            )
        with st.spinner("Listening..."):
            try:
                llm = get_llm()
                run(session.submit_answer(llm, verdict.text))
                st.session_state.usage = llm.usage.summary()
            except (LLMUnavailable, PermissionError) as exc:
                st.error(str(exc))
                return
        st.session_state.answer_nonce += 1
        if session.finished:
            st.session_state.stage = "report"
        st.rerun()

    # Past turns, without scores - the candidate must not see the scoreboard mid-interview.
    if session.turns:
        with st.expander(f"Transcript so far ({len(session.turns)} answers)"):
            for t in session.turns:
                st.markdown(f"**{t.question}**")
                st.caption(t.answer)


# --------------------------------------------------------------------------- report
def screen_report() -> None:
    session: InterviewSession = st.session_state.session

    if st.session_state.report is None:
        if not session.turns:
            st.warning("No answers were recorded, so there is nothing to score.")
            return
        with st.spinner("Writing your report..."):
            try:
                llm = get_llm()
                st.session_state.report = run(build_report(llm, session))
                st.session_state.usage = llm.usage.summary()
            except (LLMUnavailable, PermissionError) as exc:
                st.error(str(exc))
                return

    report = st.session_state.report
    st.title("Interview report")

    c1, c2 = st.columns([1, 2])
    c1.metric("Overall", f"{report.overall_score:.0f}/100")
    c2.metric("Would I move you forward?", RECOMMENDATION_LABEL.get(report.recommendation, "-"))
    st.write(report.recommendation_reason)

    if report.per_topic:
        st.subheader("By subject")
        for t in report.per_topic:
            st.markdown(f"**{t.topic}** - {t.score:.1f}/5")
            st.progress(min(t.score / 5, 1.0))
            st.caption(t.comment)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Strongest")
        for s in report.strengths:
            st.markdown(f"- {s}")
    with c2:
        st.subheader("Weakest")
        for w in report.weaknesses:
            st.markdown(f"- {w}")

    if report.unmet_requirements:
        st.subheader("Job requirements you did not demonstrate")
        for r in report.unmet_requirements:
            st.markdown(f"- {r}")

    if report.struggled_questions:
        st.subheader("Questions you struggled with")
        for q in report.struggled_questions:
            st.markdown(f"- {q}")

    if report.study_plan:
        st.subheader("Study this before the real interview")
        for i, s in enumerate(report.study_plan, 1):
            st.markdown(f"{i}. {s}")

    st.download_button(
        "Download report (Markdown)",
        data=report_markdown(report, session),
        file_name="interview_report.md",
        mime="text/markdown",
    )
    st.caption("Contact details and credentials are redacted from the download.")
    render_plan(session)


# --------------------------------------------------------------------------- main
def main() -> None:
    """Render one Streamlit rerun. `app.py` at the project root calls this."""
    init_state()
    difficulty, strategy, persona, verbosity = sidebar()
    if st.session_state.stage == "setup":
        screen_setup(difficulty, strategy, persona, verbosity)
    elif st.session_state.stage == "interview":
        screen_interview()
    else:
        screen_report()


if __name__ == "__main__":      # `streamlit run src/interview_ai/ui/app.py` also works
    main()
