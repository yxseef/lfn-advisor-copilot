"""Ask the Book widgets shared by the floating window and the full page (UI only, not a page)."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from core import llm, risk
from core.agent import (
    DEMO_QUESTIONS,
    MAIN_RISK_QUESTION,
    MAX_LIVE_QUESTIONS_PER_SESSION,
    AskAnswer,
    AskContext,
    answer_question,
    new_ask_log_entry,
)
from core.portfolios import all_tickers, build_clients

ASK_PAGE = "app_pages/4_Ask_the_Book.py"
ASK_TITLE = "Ask the Book"
NO_CLIENT = "No client selected"
LIVE_UNAVAILABLE_KEY = "live_unavailable"
# Session key of the client selector on each page, so the assistant knows who "this client" is.
CLIENT_KEYS: dict[str, str] = {
    "Risk Monitor": "rm_client",
    "Meeting Brief": "client",
    ASK_TITLE: "ask_focus_client",
}

# The floating button is a keyed container pinned with CSS. The popover body
# selector relies on Streamlit's internal test id; checked on Streamlit 1.65.
STYLE = """
<style>
.st-key-ask_fab { position: fixed; right: 1.5rem; bottom: 1.5rem; z-index: 999990; width: auto !important; }
.st-key-ask_fab button { box-shadow: 0 6px 18px rgba(15, 23, 42, 0.18); padding: 0.5rem 1.4rem; font-weight: 600; }
[data-testid="stPopoverBody"]:has(.st-key-ask_window) {
  width: min(460px, calc(100vw - 1.5rem)); max-width: calc(100vw - 1.5rem);
  max-height: min(72vh, 680px); overflow-y: auto;
}
.st-key-ask_window [data-testid="stVerticalBlock"] { gap: 0.6rem; }
.st-key-ask_window [data-testid="stButton"] button,
.st-key-ask_window [data-testid="stButton"] button * { justify-content: flex-start; text-align: left; }
.ask-banner {
  background: #f1f5f9; border-left: 3px solid #1e3a5f; border-radius: 4px; padding: 0.4rem 0.65rem;
  color: #334155; font-size: 0.78rem; line-height: 1.35; margin-bottom: 0.6rem;
}
.ask-banner strong { color: #1e3a5f; font-weight: 600; }
[class*="st-key-ask_q_"] p { color: #1f2937; font-weight: 500; }
@media (max-width: 640px) {
  .st-key-ask_fab { right: 0.75rem; bottom: 0.75rem; }
}
</style>
"""


@st.cache_data(ttl=3600, show_spinner="Loading the book…")
def load_book() -> list[risk.ClientRisk]:
    """Same book as the Risk Monitor: fictional clients, real prices, real alerts."""
    clients = build_clients()
    data = risk.load_prices(all_tickers(clients))
    return [risk.analyze_client(client, data.prices) for client in clients]


@st.cache_data
def client_names() -> dict[str, str]:
    """Name to id. Cheap: no prices are loaded to know who is selected."""
    return {client.name: client.client_id for client in build_clients()}


def init_state() -> None:
    st.session_state.setdefault("ask_history", [])
    st.session_state.setdefault("ask_log", [])
    st.session_state.setdefault("ask_live_used", 0)


def live_api_key() -> str | None:
    """The API key, unless a call failed earlier in this session: the session then stays in demo mode."""
    if st.session_state.get(LIVE_UNAVAILABLE_KEY):
        return None
    return llm.resolve_api_key()


def live_unavailable() -> bool:
    return bool(st.session_state.get(LIVE_UNAVAILABLE_KEY))


def mark_live_unavailable() -> None:
    st.session_state[LIVE_UNAVAILABLE_KEY] = True


def current_context(page: str) -> AskContext:
    key = CLIENT_KEYS.get(page)
    name = st.session_state.get(key) if key else None
    names = client_names()
    if isinstance(name, str) and name in names:
        return AskContext(page, names[name], name)
    return AskContext(page)


def ask(question: str, page: str) -> None:
    """Answer one question and store it. Runs as a widget callback, before the page reruns."""
    init_state()
    if not question or not question.strip():
        return
    history: list[AskAnswer] = [turn["answer"] for turn in st.session_state.ask_history]
    answer, screening = answer_question(
        question,
        load_book(),
        current_context(page),
        api_key=live_api_key(),
        history=history,
        live_questions_used=int(st.session_state.ask_live_used),
    )
    if answer.api_called:
        st.session_state.ask_live_used += 1
    if answer.api_failed:
        mark_live_unavailable()
    shown = "[withheld: contained personal data]" if screening.personal_data else question.strip()
    st.session_state.ask_history.append({"question": shown, "answer": answer})
    st.session_state.ask_log.append(new_ask_log_entry(answer, screening.personal_data))


def submit_from_key(key: str, page: str) -> None:
    ask(str(st.session_state.get(key) or ""), page)


def mode_line() -> str:
    if live_api_key():
        used = int(st.session_state.get("ask_live_used", 0))
        return f"Live mode · {used} of {MAX_LIVE_QUESTIONS_PER_SESSION} questions used this session"
    if live_unavailable():
        return "Demo mode · the live assistant is unavailable right now; the example questions still work"
    return "Demo mode · no API key: the example questions are answered with live figures"


def context_line(context: AskContext) -> str:
    client = context.client_name or "no client selected"
    return f"Context: {context.page} · {client}"


def example_questions(context: AskContext) -> list[str]:
    questions = list(DEMO_QUESTIONS)
    if context.client_id:
        questions.insert(0, MAIN_RISK_QUESTION)
    return questions


def render_examples(page: str, context: AskContext, prefix: str, *, stretch: bool) -> None:
    with st.container(horizontal=not stretch, gap="small"):
        for index, question in enumerate(example_questions(context)):
            st.button(
                question,
                key=f"{prefix}_example_{index}",
                on_click=ask,
                args=(question, page),
                width="stretch" if stretch else "content",
            )


def _escape(text: str) -> str:
    """Keep a dollar sign from being read as LaTeX by st.markdown."""
    return text.replace("$", "\\$")


def render_turn(
    turn: dict[str, Any], index: int, *, compact: bool, expanded: bool, page: str, latest: bool
) -> None:
    answer: AskAnswer = turn["answer"]
    prefix = "fab" if compact else "page"
    if answer.status == "demo_only":
        with st.container(key=f"ask_q_{prefix}_{index}"):
            st.caption("You asked")
            st.markdown(_escape(turn["question"]))
        with st.container(border=True):
            st.markdown(answer.text)
            if latest:
                render_examples(page, current_context(page), f"{prefix}_after_{index}", stretch=compact)
        return
    with st.container(key=f"ask_q_{prefix}_{index}"):
        st.caption("You asked")
        st.markdown(_escape(turn["question"]))
    with st.container(border=True):
        label, _, rest = answer.banner.partition(" · ")
        st.markdown(
            f'<div class="ask-banner"><strong>{html.escape(label)}</strong> · {html.escape(rest)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_escape(answer.text))
        if answer.reason and answer.status == "answered":
            st.caption(answer.reason)
        if answer.figures_to_verify:
            st.caption("To verify — not found in the tool results: " + ", ".join(answer.figures_to_verify))
        if answer.advice_rewritten:
            st.caption(
                f"The output filter rewrote {answer.advice_rewritten} sentence(s) that read as an instruction to transact."
            )
        if answer.redactions:
            st.caption(f"The output filter removed {answer.redactions} sentence(s) with an email, an IBAN or an AVS number.")
        with st.expander(f"Tools used ({len(answer.tool_calls)})", expanded=expanded):
            if not answer.tool_calls:
                st.caption("No tool was called for this answer.")
            for number, call in enumerate(answer.tool_calls, start=1):
                status = "" if call.ok else " · returned an error"
                st.markdown(f"**{number}. `{call.name}`**{status}")
                st.caption("Parameters")
                st.json(call.params or {}, expanded=True)
                st.caption("Result")
                st.json(call.result, expanded=1 if compact else 2)
            st.caption(
                f"{answer.mode.capitalize()} mode · {answer.created_at}"
                + (f" · {answer.model_calls} model call(s)" if answer.model_calls else " · no model call")
            )
        if compact:
            st.page_link(ASK_PAGE, label="Open full view")


def render_fab(page: str) -> None:
    """The floating "Ask" button and its compact chat window, for every page but the full view."""
    init_state()
    context = current_context(page)
    with st.container(key="ask_fab"):
        with st.popover("Ask", type="primary", key="ask_popover"):
            with st.container(key="ask_window"):
                st.markdown("**Ask the Book**")
                st.caption(f"{mode_line()}  \n{context_line(context)}")
                st.chat_input(
                    "Ask about the book",
                    key="ask_fab_input",
                    on_submit=submit_from_key,
                    args=("ask_fab_input", page),
                    max_chars=500,
                )
                history = st.session_state.ask_history
                last = history[-1] if history else None
                if last:
                    render_turn(last, len(history) - 1, compact=True, expanded=False, page=page, latest=True)
                if last and last["answer"].status == "demo_only":
                    st.page_link(ASK_PAGE, label="Open full view")
                else:
                    st.markdown("**Example questions**")
                    render_examples(page, context, "fab", stretch=True)
                if not history:
                    st.page_link(ASK_PAGE, label="Open full view")
