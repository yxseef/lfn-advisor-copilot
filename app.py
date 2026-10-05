"""LFN Advisor Copilot: entrypoint. The sidebar label comes from st.Page, not the filename."""

from __future__ import annotations

import streamlit as st

from core import APP_NAME, AUTHOR_CREDIT, DISCLAIMER


def home() -> None:
    st.warning(DISCLAIMER, icon="⚠️")
    st.title(APP_NAME)
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
| **LFN Market Map** | Maps law, fiduciary and notary firms in Geneva and Vaud for business development. | Planned |
"""
    )

    st.page_link("pages/1_Risk_Monitor.py", label="Open the Risk Monitor", icon="📊")
    st.page_link("pages/2_Meeting_Brief.py", label="Open the Meeting Brief", icon="📝")

    st.divider()
    st.caption(
        f"{DISCLAIMER} All clients and portfolios are invented. No affiliation with any bank. "
        "Market data from Yahoo Finance, for illustration only."
    )
    st.markdown(AUTHOR_CREDIT)


st.set_page_config(page_title=APP_NAME, layout="wide")
navigation = st.navigation(
    [
        st.Page(home, title="Home", default=True, icon="🏠"),
        st.Page("pages/1_Risk_Monitor.py", title="Risk Monitor", icon="📊", url_path="Risk_Monitor"),
        st.Page("pages/2_Meeting_Brief.py", title="Meeting Brief", icon="📝", url_path="Meeting_Brief"),
    ]
)
navigation.run()
