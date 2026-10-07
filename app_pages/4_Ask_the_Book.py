"""Ask the Book: full conversation view of the read-only assistant (UI only)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_pages.ask_ui import (
    ASK_TITLE,
    NO_CLIENT,
    STYLE,
    client_names,
    context_line,
    current_context,
    init_state,
    mode_line,
    render_examples,
    render_turn,
    submit_from_key,
)
from app_pages.guide import page_intro
from core import APP_NAME, AUTHOR_CREDIT, DISCLAIMER, llm
from core.agent import (
    AGENT_PROMPT_VERSION,
    BANNER,
    MAX_LIVE_QUESTIONS_PER_SESSION,
    MAX_MODEL_CALLS_PER_QUESTION,
    TOOL_SPECS,
    build_agent_system_prompt,
)

st.set_page_config(page_title=f"{ASK_TITLE} · {APP_NAME}", layout="wide")
st.html(STYLE)
init_state()

st.warning(DISCLAIMER)
st.title(ASK_TITLE)
page_intro(
    "Ask about the whole book in plain English; answers cite figures from the Risk Monitor's own functions.",
    [
        "Optionally put a **client in focus**, so that “this client” in a question means that client.",
        "Click an **example question**, or type your own in the box at the bottom of the page.",
        "Open **Tools used** under an answer to see which functions ran and the figures they returned.",
    ],
)

if llm.resolve_api_key():
    st.caption(
        f"Live mode: the model chooses the tools and writes the answer. "
        f"{int(st.session_state.ask_live_used)} of {MAX_LIVE_QUESTIONS_PER_SESSION} questions used this session."
    )
else:
    st.info(
        "Demo mode — no API key is configured, so no model is called. The example questions below have "
        "pre-written answers, and every figure in them is computed now by the same tools a live model would call."
    )

with st.container(border=True):
    st.markdown("**Guardrails — always on**")
    st.markdown(
        f"""
1. **Read-only.** Five tools read the Risk Monitor. None can trade, edit data or contact anyone; such requests are refused.
2. **Figures from tools.** The model never computes. A figure that no tool returned is marked "to verify".
3. **Input filter.** Emails, IBANs, AVS numbers, unknown names and attempts to override the rules are blocked before any call.
4. **No recommendation.** Discussion topics only, never an instruction to buy or sell.
5. **Banner, cap and journal.** Every answer is labelled "{BANNER}". Live mode stops after {MAX_LIVE_QUESTIONS_PER_SESSION} questions per session. Every question is logged below.
"""
    )

names = list(client_names())
st.selectbox(
    "Client in focus",
    [NO_CLIENT, *names],
    key="ask_focus_client",
    help="Optional. With a client in focus, “this client” in a question means that client.",
)
context = current_context(ASK_TITLE)
st.caption(f"{mode_line()} · {context_line(context)}")

st.markdown("**Example questions**")
render_examples(ASK_TITLE, context, "page", stretch=False)

st.subheader("Conversation")
history = st.session_state.ask_history
if not history:
    st.caption("No question yet. Pick an example or type a question in the box at the bottom of the page.")
for index, turn in enumerate(history):
    latest = index == len(history) - 1
    render_turn(turn, index, compact=False, expanded=latest, page=ASK_TITLE, latest=latest)
if history and st.button("Clear conversation", key="ask_clear"):
    st.session_state.ask_history = []
    st.rerun()

st.divider()
st.subheader("Question journal")
st.caption(
    "One row per question, kept in this browser session. It stores the screened question (fictional clients "
    "appear as their id), the tools called, the mode and the status. Not the answer, not the tool results, not the key."
)
if st.session_state.ask_log:
    log = pd.DataFrame(st.session_state.ask_log).iloc[::-1]
    st.dataframe(
        log.rename(
            columns={
                "timestamp": "Timestamp (UTC)",
                "question": "Question (screened)",
                "tools": "Tools called",
                "mode": "Mode",
                "status": "Status",
                "reason": "Reason",
                "prompt_version": "Prompt",
            }
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.caption("No question in this session yet.")

st.divider()
st.subheader("How this agent works & its limits")
st.markdown(
    f"""
**Who does what.** The model reads the question and decides which function to call, with which parameters.
The application runs the function on the Risk Monitor book and sends the result back. The model then writes
the answer from those results. It does not do arithmetic: a total, an average or an extrapolation it writes
on its own comes back marked "to verify". At most {MAX_MODEL_CALLS_PER_QUESTION} model calls per question.

**Read-only by construction.** The model can only ask for the functions in the list below. None of them writes,
trades or sends anything, so a request to sell, email or change a limit has no tool to reach. The input
filter refuses such requests before the model is called, and the answer explains why.

**Context.** Each question carries the open page and the client selected on it. On a client's page,
"this client" or "their main risk" means that client.

**What it does not guarantee.** The filters are pattern-based. They can miss an unusual phrasing, and they can
block a harmless question that looks like a name or an order. The figure check is lexical: it confirms that
a number came from a tool, not that it was used for the right concept. A sentence without a figure can still be
wrong. Demo mode answers only the examples. Prompt version **{AGENT_PROMPT_VERSION}**.
"""
)
with st.expander("The five tools"):
    for spec in TOOL_SPECS:
        st.markdown(f"**`{spec.name}`** — {spec.description}")
with st.expander(f"System prompt (version {AGENT_PROMPT_VERSION})"):
    st.code(build_agent_system_prompt(), language="text")

st.divider()
st.caption(f"{DISCLAIMER} All clients are invented. No affiliation with any bank.")
st.markdown(AUTHOR_CREDIT)

st.chat_input(
    "Ask about the book",
    key="ask_page_input",
    on_submit=submit_from_key,
    args=("ask_page_input", ASK_TITLE),
    max_chars=500,
)
