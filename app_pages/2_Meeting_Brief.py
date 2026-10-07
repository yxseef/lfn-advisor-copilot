"""Meeting Brief: a guarded pre-meeting draft for one fictional LFN client (UI only)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_pages.ask_ui import live_api_key, live_unavailable, mark_live_unavailable
from app_pages.guide import page_intro
from core import APP_NAME, AUTHOR_CREDIT, DISCLAIMER
from core.llm import (
    BANNER,
    DRAFT_NOTICE,
    MAX_LIVE_GENERATIONS_PER_SESSION,
    MAX_NOTE_CHARS,
    MEETING_TYPES,
    PROMPT_VERSION,
    BriefDraft,
    ExportNotReviewed,
    build_system_prompt,
    client_facts,
    export_markdown,
    export_pdf,
    generate_meeting_brief,
    mark_log_reviewed,
    new_log_entry,
    screen_advisor_note,
    utc_now_str,
)
from core.portfolios import all_tickers, build_clients
from core.risk import analyze_client, load_prices

st.set_page_config(page_title=f"Meeting Brief · {APP_NAME}", layout="wide")

SEVERITY_BOX = {"breach": st.error, "warning": st.warning, "info": st.info}


@st.cache_data(ttl=3600, show_spinner="Loading market data and computing risk…")
def load_book() -> tuple[list, str, list[str], pd.Timestamp]:
    """Same book as the Risk Monitor: fictional clients, real prices, real alerts."""
    clients = build_clients()
    data = load_prices(all_tickers(clients))
    results = [analyze_client(client, data.prices) for client in clients]
    return results, data.source, data.missing, data.prices.index[-1]


def _init_state() -> None:
    st.session_state.setdefault("brief_log", [])
    st.session_state.setdefault("live_calls", 0)
    st.session_state.setdefault("reviewed_at", {})
    st.session_state.setdefault("current_draft", None)


def _render_draft(draft: BriefDraft, reviewed: bool, reviewed_at: str | None) -> None:
    facts = draft.facts
    brief = draft.brief
    displays = facts["displays"]

    if reviewed:
        st.warning(
            f"{BANNER}. {DRAFT_NOTICE} An advisor has marked this draft as reviewed. "
            "The file still carries this banner.",
        )
    else:
        st.warning(
            f"{BANNER}. {DRAFT_NOTICE} "
            "Export stays locked until you press Reviewed by advisor.",
        )
    if draft.fallback_reason:
        st.info(draft.fallback_reason)
    elif draft.mode == "demo":
        st.caption(
            "Demo mode: the qualitative text comes from data/demo_briefs.json. "
            "Figures and alerts were computed by the Risk Monitor just now. No model was called."
        )
    else:
        st.caption(
            "Live mode: the qualitative text comes from the model, after the output checks. "
            "Figures and alerts still come from the Risk Monitor, not from the model."
        )
    if draft.note_accepted and not draft.advisor_note_sent:
        st.caption(
            "A note was entered and it passed the personal-data check. "
            "Demo mode did not use it, so the pre-generated text is unchanged."
        )
    if draft.advisor_note_sent:
        st.caption(
            "The note was sent as context only. It is not a source of figures, "
            "and it is not stored in the journal or in the export."
        )

    st.subheader(facts["name"])
    st.markdown(
        f"**{facts['meeting_label']}** · {facts['client_type']} · "
        f"{facts['account_type']} account · **{facts['risk_profile']}** profile · "
        f"prompt {draft.prompt_version} · generation `{draft.generation_id}`"
    )

    st.markdown("**Situation**")
    st.write(brief.situation)

    st.markdown("**Figures computed by the Risk Monitor**")
    st.caption("Written by the application from the risk engine. The model does not write these numbers.")
    metrics = st.columns(4)
    metrics[0].metric("AUM", displays["aum"])
    metrics[1].metric("Volatility (ann.)", displays["volatility"])
    metrics[2].metric("VaR 95% 1-day", displays["var_1d"])
    metrics[3].metric("Equities", displays["equity"])
    if facts.get("lombard") and displays.get("loan"):
        credit = st.columns(3)
        credit[0].metric("Lombard loan", displays["loan"])
        credit[1].metric("Usage", displays["usage"])
        credit[2].metric("LTV", displays["ltv"])

    st.markdown("**Risk points**")
    st.caption("Copied from the Risk Monitor alerts for this client. The model is not allowed to rewrite them.")
    if not facts["alerts"]:
        st.success(facts["risk_points"][0])
    else:
        for alert in facts["alerts"]:
            SEVERITY_BOX[alert["severity"]](f"**{alert['category']}** — {alert['message']}")

    with st.expander("Largest positions and stress tests"):
        for position in facts["positions_top"]:
            st.markdown(
                f"- {position['name']} ({position['asset_class']}): "
                f"{position['weight_display']}, {position['value_display']}"
            )
        st.markdown("**Stress tests**")
        for row in facts["stress"]:
            st.markdown(f"- {row['scenario']}: {row['pnl_display']} ({row['pnl_pct_display']})")

    st.markdown("**Probable needs**")
    st.markdown("\n".join(f"- {item}" for item in brief.probable_needs))

    st.markdown("**Banking solutions**")
    st.caption("Generic product families to discuss. Not instructions to transact.")
    for solution in brief.banking_solutions:
        st.markdown(f"**{solution.category}**")
        st.write(solution.why_discuss)

    st.markdown("**Questions to ask**")
    st.markdown("\n".join(f"- {item}" for item in brief.questions_to_ask))

    st.markdown("**Compliance**")
    st.markdown("\n".join(f"- {item}" for item in brief.compliance_points))

    st.caption(
        f"Output checks on this draft: {len(draft.figures_to_verify)} figure(s) marked to verify, "
        f"{draft.advice_rewritten} instruction(s) rewritten, "
        f"{draft.sensitive_redactions} identifier(s) removed."
    )
    if draft.figures_to_verify:
        st.warning(
            "To verify — these figures were not in the client data: "
            + ", ".join(draft.figures_to_verify)
        )
    if draft.advice_rewritten:
        st.warning(
            f"The output filter rewrote {draft.advice_rewritten} sentence(s) that read as an instruction "
            "to transact. They are prefixed “Discussion topic only, not an order”."
        )
    if draft.sensitive_redactions:
        st.warning(
            f"The output filter removed {draft.sensitive_redactions} sentence(s) that contained "
            "an email, an IBAN or an AVS number."
        )

    st.divider()
    if st.button(
        "Reviewed by advisor",
        type="primary",
        key="mark_reviewed",
        help="Records that you have read this draft. It does not verify your identity.",
    ):
        stamp = utc_now_str()
        st.session_state.reviewed_at[draft.generation_id] = stamp
        st.session_state.brief_log = mark_log_reviewed(
            st.session_state.brief_log, draft.generation_id, stamp
        )
        st.rerun()

    if reviewed:
        st.success(
            f"Reviewed by advisor{f' at {reviewed_at}' if reviewed_at else ''}. "
            "Export is unlocked for this draft."
        )
    else:
        st.info("Export stays locked until you press Reviewed by advisor. Reading the draft is not a review.")

    try:
        markdown = export_markdown(draft, reviewed=reviewed, reviewed_at=reviewed_at) if reviewed else ""
        pdf = export_pdf(draft, reviewed=reviewed, reviewed_at=reviewed_at) if reviewed else b""
    except ExportNotReviewed:
        markdown = ""
        pdf = b""
        reviewed = False

    slug = f"meeting-brief-{facts['client_id']}-{facts['meeting_type']}"
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download Markdown",
            data=markdown,
            file_name=f"{slug}.md",
            mime="text/markdown",
            disabled=not reviewed,
            key="download_md",
        )
    with right:
        st.download_button(
            "Download PDF",
            data=pdf,
            file_name=f"{slug}.pdf",
            mime="application/pdf",
            disabled=not reviewed,
            key="download_pdf",
        )


def _how_it_works() -> None:
    st.subheader("How this AI works & its limits")
    st.markdown(
        f"""
**What it is.** The model drafts talking points for one meeting. The advisor decides.
The draft is not a recommendation, not an order, and not a completed KYC or credit file.
Prompt version **{PROMPT_VERSION}**.

**The system prompt is an instruction, not the control.** It tells the model to use only the
selected client's facts, to avoid figures where a qualitative sentence will do, to repeat the
Risk Monitor alerts, and to stay inside a fixed list of generic banking solutions
(Lombard credit, wealth planning, a consignment account, and the others in the list).
Every check below is code. It runs on the pre-generated text and on model text alike,
whether or not the model complies.

**1. Input.** `client_facts` builds one JSON object from the client you select and from that
client's Risk Monitor result. No other client is included. The free-text note is optional.
Before a brief is generated, `screen_advisor_note` rejects an email address, an IBAN, a Swiss
AVS/AHV number, a personal name, and a note that tries to override the instructions.
A rejected note is not sent, not stored in the journal, and not written into the brief.
In demo mode a note that passes is still ignored, so the stored text cannot be altered.
In live mode the note is context only: it is not added to the figure allow-list, so a figure
copied from it comes back marked "to verify".

**2. Output.** The model must return JSON matching `MeetingBrief`. Pydantic checks the schema,
and a field the schema does not define is dropped. `apply_output_guardrails` then replaces
`risk_points` with the real alerts, so the model cannot rewrite them. The figures block is
written by the application from the risk engine. Every number in the model's own sentences
is compared with the numbers in the facts (`is_grounded`). A magnitude that is not there is
marked "(to verify)" and listed. An amount matches within one percent. A percent matches
within about half a percentage point. A small count has to match almost exactly.
The check is lexical: it does not know whether a number was used for the right concept,
and it does not check the sign. A sentence that contains an email, an IBAN or an AVS number
is removed. Names are filtered on the way in, not on the way out, because the brief is
supposed to name the selected fictional client.

**3. No personalised investment recommendation.** Solutions are generic families. The prompt
forbids an instruction to buy, sell, allocate or switch. A second filter rewrites any sentence
that still reads that way and prefixes it with "Discussion topic only, not an order".

**4. Human review.** The banner "{BANNER}" stays on the draft and on the file.
Markdown and PDF stay disabled until you press "Reviewed by advisor".
That press records a claim that a person read the draft. It does not authenticate who pressed it.
Copying the text off the screen is still possible: the gate stops the export buttons, it is not a lock on the words.

**5. Journal.** Each generation adds a row: time (UTC), prompt version, client, meeting type,
mode, and review status. The row does not contain the brief and does not contain the note.
The journal lives in this browser session and disappears when the session ends.
A production log would be stored outside the browser, append-only.

**6. Demo mode and the cap.** With no key in Streamlit secrets, the environment, or a local
`.env` file, the page never calls a model. It joins `data/demo_briefs.json` to the figures
computed now, so the numbers stay the same ones as the Risk Monitor. The stored file has no
amounts in it, which is why a price move cannot make the text go stale.
With a key, live calls are capped at {MAX_LIVE_GENERATIONS_PER_SESSION} per session.
After that the API is not called and the pre-generated brief is shown.
If a live call fails, or the JSON does not match the schema, the same fallback is used
and the screen says so.

**What this does not guarantee.** The name filter misses unusual forms and can block a
harmless title such as two capitalised words. The number check can accept a figure that
appears somewhere in the file and is then used in the wrong sentence. The model can still
be wrong in a sentence that contains no figure. Review is a gate, not a signature.
None of this is investment advice, and every client is fictional.
"""
    )
    with st.expander(f"System prompt (version {PROMPT_VERSION})"):
        st.code(build_system_prompt(), language="text")


_init_state()

st.warning(DISCLAIMER)
st.title("Meeting Brief")
page_intro(
    "Prepare a pre-meeting draft for one client: discussion topics for the advisor, never an order.",
    [
        "Choose a **client** and a **meeting** type. A short context note is optional.",
        "Press **Generate brief**. Figures and alerts come from the Risk Monitor, the text from the AI.",
        "Read the draft, then press **Reviewed by advisor** to unlock the Markdown and PDF export.",
        "The **Generation journal** below records every draft produced in this session.",
    ],
)

results, source, missing, as_of = load_book()
st.caption(f"Risk figures: **{source}** · last close **{as_of:%d %b %Y}** · same engine as the Risk Monitor.")
if missing:
    st.warning(f"No price data for: {', '.join(missing)}. Affected positions are excluded from the figures.")

with st.container(border=True):
    st.markdown("**Guardrails — always on**")
    st.markdown(
        f"""
1. **Input.** Only the fictional client you select is used. The note is rejected if it contains a name, an email, an IBAN or an AVS number.
2. **Output.** The draft is validated JSON. Alerts and figures come from the Risk Monitor. Any other figure is marked "to verify".
3. **No order.** Solutions are generic discussion topics (Lombard credit, wealth planning, a consignment account, …).
4. **Human review.** The draft is labelled "{BANNER}". Export stays off until you press "Reviewed by advisor".
5. **Journal and cap.** Every generation is logged. Live calls stop after {MAX_LIVE_GENERATIONS_PER_SESSION} per session. With no API key, a pre-generated brief is used.
"""
    )

by_name = {result.client.name: result for result in results}
selectors = st.columns(2)
with selectors[0]:
    client_name = st.selectbox(
        "Client",
        list(by_name),
        format_func=lambda name: f"{name}  ·  {by_name[name].status.upper()}",
        key="client",
    )
with selectors[1]:
    meeting_type = st.selectbox(
        "Meeting",
        list(MEETING_TYPES),
        format_func=lambda key: MEETING_TYPES[key],
        key="meeting_type",
    )

result = by_name[client_name]
facts = client_facts(result, meeting_type)
note = st.text_area(
    "Optional note for this meeting",
    max_chars=MAX_NOTE_CHARS,
    height=100,
    key="advisor_note",
    placeholder="Context only, for example the purpose of the meeting. Do not type names, emails, IBANs or AVS numbers.",
    help="The client is chosen above. This box is screened before anything is generated.",
)

api_key = live_api_key()
used = int(st.session_state.live_calls)
if api_key:
    st.caption(
        f"Live mode available. Generations this session: {used} of {MAX_LIVE_GENERATIONS_PER_SESSION}. "
        "The cap protects the API key."
    )
elif live_unavailable():
    st.caption(
        "Demo mode: the live AI is unavailable right now, so briefs use the pre-written text. "
        "Figures and alerts are still computed by the Risk Monitor."
    )
else:
    st.caption(
        "Demo mode: no API key in Streamlit secrets, the environment, or .env. "
        f"A live key would be limited to {MAX_LIVE_GENERATIONS_PER_SESSION} generations per session."
    )

with st.expander("Payload for the selected client"):
    st.caption(
        "This is the only client data a generation can use. "
        "The note is not part of it, and it is not a source of figures."
    )
    st.json(facts)

if st.button("Generate brief", type="primary", key="generate_brief"):
    findings = screen_advisor_note(note)
    if findings:
        for finding in findings:
            st.error(finding.message)
    else:
        with st.spinner("Preparing the brief…"):
            draft = generate_meeting_brief(
                facts,
                api_key=api_key,
                advisor_note=note,
                live_calls_already=used,
            )
        if draft.api_called:
            st.session_state.live_calls += 1
        if draft.api_failed:
            mark_live_unavailable()
        st.session_state.current_draft = draft
        st.session_state.brief_log.append(new_log_entry(draft))

current = st.session_state.current_draft
if isinstance(current, BriefDraft):
    if current.facts["client_id"] != facts["client_id"] or current.facts["meeting_type"] != meeting_type:
        st.info(
            f"The draft below was generated for {current.facts['name']} · "
            f"{current.facts['meeting_label']}. Generate again to replace it."
        )
    stamp = st.session_state.reviewed_at.get(current.generation_id)
    _render_draft(current, reviewed=stamp is not None, reviewed_at=stamp)

st.divider()
st.subheader("Generation journal")
st.caption(
    "This journal lives in the browser session. It records that a review was claimed; "
    "it does not verify who clicked, and it does not store the brief or the note."
)
if st.session_state.brief_log:
    frame = pd.DataFrame(st.session_state.brief_log)
    columns = {
        "timestamp": "Timestamp (UTC)",
        "prompt_version": "Prompt version",
        "client": "Client",
        "meeting_type": "Meeting",
        "mode": "Mode",
        "review_status": "Review status",
        "reviewed_at": "Reviewed at",
        "generation_id": "Id",
    }
    ordered = [name for name in columns if name in frame.columns]
    st.dataframe(
        frame.iloc[::-1][ordered].rename(columns=columns),
        hide_index=True,
        width="stretch",
    )
else:
    st.caption("No generation in this session yet.")

st.divider()
_how_it_works()

st.divider()
st.caption(f"{DISCLAIMER} The same alerts are on the Risk Monitor page.")
st.markdown(AUTHOR_CREDIT)
