"""LFN Advisor Copilot: entrypoint. The sidebar label comes from st.Page, not the filename."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core import APP_NAME, AUTHOR_CREDIT, DISCLAIMER

SCREENSHOTS = Path(__file__).resolve().parent / "docs" / "screenshots"
TAGLINE = (
    "A prototype assistant for LFN client advisors: it monitors portfolio risk, "
    "prepares meetings with guardrailed AI, and helps prioritise prospects."
)


def home() -> None:
    st.warning(DISCLAIMER, icon="⚠️")
    st.title(APP_NAME)
    st.markdown(f"#### {TAGLINE}")
    st.markdown(
        "*An independent student prototype for client advisors serving lawyers, fiduciaries and notaries (LFN).*"
    )
    st.markdown(AUTHOR_CREDIT)

    st.subheader("The problem")
    st.markdown(
        """
LFN clients are professionals with two sets of needs: their own (private wealth, firm treasury,
Lombard credit) and those of the clients they represent (estates, asset structures, property deals).
An advisor covering dozens of such relationships must spot risk issues early and prepare every
meeting thoroughly, with limited time.
"""
    )

    st.subheader("The solution")
    st.markdown(
        """
A small toolkit that automates the monitoring work and leaves judgement to the advisor:

| Module | What it does | Status |
|---|---|---|
| **Risk Monitor** | Values fictional portfolios with real market data; volatility, VaR, concentration, Lombard LTV, profile compliance, stress tests and automatic alerts. | Available |
| **Meeting Brief (AI)** | Structured pre-meeting briefing for one fictional client, with guardrails and human review. | Available |
| **LFN Market Map** | Maps law, fiduciary and notary firms in Geneva and Vaud and ranks prospects with a transparent score. | Available (fictional example data) |
"""
    )
    st.info(
        "**Market Map data are a fictional example.** Access to the federal commercial register API (Zefix) "
        "is restricted: it requires credentials issued by the Federal Registry of Commerce and has no "
        "industry (NOGA) code. So the public demo ranks invented firms, all named “Exemple…”, placed on "
        "real Geneva and Vaud communes. The scoring and the map work the same on a real register extract.",
        icon="ℹ️",
    )

    st.subheader("The modules")
    modules = [
        ("risk_monitor.png", "Risk Monitor — book overview with alerts by severity.", "app_pages/1_Risk_Monitor.py", "Open the Risk Monitor", "📊"),
        ("meeting_brief.png", "Meeting Brief — demo draft, risk-engine figures.", "app_pages/2_Meeting_Brief.py", "Open the Meeting Brief", "📝"),
        ("market_map.png", "Market Map — fictional firms on commune centres.", "app_pages/3_Market_Map.py", "Open the Market Map", "🗺️"),
    ]
    for column, (image, caption, page, label, icon) in zip(st.columns(3), modules):
        with column:
            path = SCREENSHOTS / image
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")
            st.page_link(page, label=label, icon=icon)

    st.subheader("Responsible AI")
    st.markdown(
        """
The Meeting Brief is the only module that uses a language model. Its guardrails are on by default:

- **Only fictional data goes in.** The prompt holds one fictional client from the Risk Monitor. The advisor's free-text note is blocked if it contains a name, an email, an IBAN or a Swiss AVS number.
- **Figures come from the risk engine.** AUM, volatility, VaR and alerts are written by the application. Any other number the model writes is marked “(to verify)”.
- **Discussion topics, not orders.** Banking solutions come from a closed list of generic product families, and sentences that read like a trading instruction are rewritten.
- **A human signs off.** The draft is labelled “AI-generated draft”, and export to PDF or Markdown stays locked until the advisor marks it as reviewed.
- **No key, no cost.** Without an API key the page uses pre-written text and makes no model call. With a key, live calls are capped at five per session and each one is logged without the brief or the note.
"""
    )

    st.subheader("Tech stack")
    st.markdown(
        """
Python · Streamlit (multipage) · pandas · NumPy · Plotly · yfinance (Yahoo Finance prices, CSV fallback) ·
Anthropic Claude API through one wrapper (optional) · pydantic (output schema) · fpdf2 (PDF export) ·
swisstopo geo.admin.ch (commune geocoding) · pytest · Streamlit Community Cloud.
"""
    )

    st.divider()
    st.caption(
        f"{DISCLAIMER} All clients, portfolios and Market Map firms are invented. No affiliation with any bank. "
        "Market data from Yahoo Finance, for illustration only."
    )
    st.markdown(AUTHOR_CREDIT)


st.set_page_config(page_title=APP_NAME, layout="wide")
navigation = st.navigation(
    [
        st.Page(home, title="Home", default=True, icon="🏠"),
        st.Page("app_pages/1_Risk_Monitor.py", title="Risk Monitor", icon="📊", url_path="Risk_Monitor"),
        st.Page("app_pages/2_Meeting_Brief.py", title="Meeting Brief", icon="📝", url_path="Meeting_Brief"),
        st.Page("app_pages/3_Market_Map.py", title="Market Map", icon="🗺️", url_path="Market_Map"),
    ]
)
navigation.run()
