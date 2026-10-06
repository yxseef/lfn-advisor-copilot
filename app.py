"""LFN Advisor Copilot: entrypoint. Top navigation, shared style, and the Home page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_pages.ask_ui import ASK_TITLE, render_fab
from app_pages.ask_ui import STYLE as ASK_STYLE
from app_pages.guide import STYLE as GUIDE_STYLE
from app_pages.guide import render_tour_banner, render_welcome
from core import APP_NAME, AUTHOR_CREDIT, DISCLAIMER

ROOT = Path(__file__).resolve().parent
SCREENSHOTS = ROOT / "docs" / "screenshots"
WORDMARK = ROOT / "assets" / "wordmark.svg"
TAGLINE = (
    "A prototype assistant for LFN client advisors: it monitors portfolio risk, "
    "prepares meetings with guardrailed AI, answers questions about the book, and helps prioritise prospects."
)
WHY_I_BUILT_THIS: str | None = (
    "Internship applications often ask candidates to show how AI can improve outcomes. Rather than describe it "
    "in a cover letter, I decided to build it. I chose one specific desk, advisors serving lawyers, fiduciaries "
    "and notaries, and asked a simple question: what slows them down, and what could a tool do about it without "
    "taking judgement away from them? This prototype is my answer: real risk analytics, an AI that can read but "
    "never act, and every number traceable to its source."
)

# Applied to every page because the entrypoint runs before each page.
STYLE = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 4.5rem; padding-bottom: 4rem; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 1.1rem; }
h1 { letter-spacing: -0.01em; margin-bottom: 0.25rem; }
h2, h3 { margin-top: 0.75rem; }
[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 0.9rem 1rem;
}
[data-testid="stMetricLabel"] p { color: #64748b; font-size: 0.8rem; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; color: #1f2937; }
[data-testid="stHeader"] { border-bottom: 1px solid #e2e8f0; }
[data-testid="stHeader"] .rc-overflow { justify-content: flex-end; }
[data-testid="stTopNavLink"] { background: transparent !important; border-radius: 0; padding-left: 0.85rem; padding-right: 0.85rem; }
[data-testid="stTopNavLink"]:hover { color: #1e3a5f; }
[data-testid="stTopNavLink"][aria-current="page"] {
  color: #1e3a5f;
  box-shadow: inset 0 -2px 0 #1e3a5f;
}
[data-testid="stTopNavLink"][aria-current="page"] p { font-weight: 600; }
[data-testid="stImage"] img { border: 1px solid #e2e8f0; border-radius: 6px; }
</style>
"""


def home() -> None:
    st.warning(DISCLAIMER)
    st.title(APP_NAME)
    st.markdown(f"**{TAGLINE}**")
    st.markdown(AUTHOR_CREDIT)
    render_welcome()

    st.subheader("What it does")
    st.markdown(
        """
LFN advisors cover dozens of lawyers, fiduciaries and notaries, each with private wealth, a firm account and
client mandates. The copilot does the monitoring and the preparation; the advisor keeps the judgement.

| Module | For the advisor | Status |
|---|---|---|
| **Risk Monitor** | Fictional portfolios at today's prices: VaR, Lombard LTV, profile limits, stress tests, alerts. | Available |
| **Meeting Brief** | A pre-meeting draft for one client. AI writes the text, the risk engine the figures, the advisor approves. | Available |
| **Ask the Book** | Questions about the whole book in plain English. The AI picks read-only risk functions and cites their figures; it cannot trade or contact anyone. | Available (demo without a key) |
| **Market Map** | Law, fiduciary and notary firms in Geneva and Vaud, ranked by a transparent prospect score. | Available |
"""
    )
    st.caption("Ask the Book is also one click away on every page: the Ask button at the bottom right.")
    st.info(
        "The Market Map uses fictional firms. The federal commercial register API (Zefix) requires credentials "
        "and has no industry code, so the public demo ranks invented firms placed on real communes."
    )

    st.subheader("The modules")
    modules = [
        ("risk_monitor.png", "Risk Monitor", "app_pages/1_Risk_Monitor.py"),
        ("meeting_brief.png", "Meeting Brief", "app_pages/2_Meeting_Brief.py"),
        ("ask_the_book.png", "Ask the Book", "app_pages/4_Ask_the_Book.py"),
        ("market_map.png", "Market Map", "app_pages/3_Market_Map.py"),
    ]
    for row in (modules[:2], modules[2:]):
        for column, (image, label, page) in zip(st.columns(2, gap="medium"), row):
            with column:
                path = SCREENSHOTS / image
                if path.exists():
                    st.image(str(path), width="stretch")
                st.page_link(page, label=f"Open {label}")

    st.subheader("Why I built this")
    if WHY_I_BUILT_THIS:
        st.markdown(WHY_I_BUILT_THIS)
    else:
        st.caption("[Placeholder — text to be provided by Yousif Bag.]")

    st.subheader("Responsible AI")
    st.markdown(
        """
- Only the selected fictional client goes into the prompt; notes with names, emails, IBAN or AVS numbers are blocked.
- Figures come from the risk engine. Any other number the model writes is marked "(to verify)".
- Discussion topics, not orders: a closed list of product families, and trading-style sentences are rewritten.
- Export stays locked until the advisor marks the draft as reviewed.
- Without an API key the page runs on pre-written text; with a key, five calls per session at most.
- Ask the Book is read-only: the model chooses among five functions that read the risk engine, never computes a
  figure itself, and refuses orders, edits and contact requests. Every answer is labelled AI-generated and logged.
"""
    )

    st.subheader("Tech stack")
    st.markdown(
        "Python · Streamlit · pandas · NumPy · Plotly · yfinance · Anthropic API with tool use (optional) · "
        "pydantic · fpdf2 · swisstopo geo.admin.ch · pytest"
    )

    st.divider()
    st.caption(f"{DISCLAIMER} All clients and firms are invented. No affiliation with any bank.")
    st.markdown(AUTHOR_CREDIT)


st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="collapsed")
st.logo(str(WORDMARK), size="large")
st.html(STYLE)
st.html(ASK_STYLE)
st.html(GUIDE_STYLE)
navigation = st.navigation(
    [
        st.Page(home, title="Home", default=True),
        st.Page("app_pages/1_Risk_Monitor.py", title="Risk Monitor", url_path="Risk_Monitor"),
        st.Page("app_pages/2_Meeting_Brief.py", title="Meeting Brief", url_path="Meeting_Brief"),
        st.Page("app_pages/4_Ask_the_Book.py", title=ASK_TITLE, url_path="Ask_the_Book"),
        st.Page("app_pages/3_Market_Map.py", title="Market Map", url_path="Market_Map"),
    ],
    position="top",
)
render_tour_banner(navigation.title)
navigation.run()
# After the page, so the selected client is known. The full view has its own input.
if navigation.title != ASK_TITLE:
    render_fab(navigation.title)
