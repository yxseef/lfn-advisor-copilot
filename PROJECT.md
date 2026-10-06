# LFN Advisor Copilot — Project Context

> Context file for the AI coding agent. Read this before every task.

## Purpose
A portfolio project built to support an application to a wealth-management internship (LFN segment, Geneva).
It shows that the author understands the Client Advisor job and can use AI **experimentally and responsibly** to improve outcomes and efficiency.

**LFN = Lawyers, Fiduciaries, Notaries.** In Swiss private banking, LFN teams advise these professionals on their own private and corporate needs (firm accounts, escrow/consignment accounts, credit, wealth planning) and on the needs of the clients they represent (estates and successions, asset structures, property transactions).

## Hard rules
- **All client data is fictional.** Never use, ask for, or simulate real client data.
- **No bank branding.** No logos, names, or anything suggesting an official tool of any real bank. The app is called "LFN Advisor Copilot", an independent student prototype.
- Show a visible disclaimer on every page: *"Student prototype — fictional data — not investment advice."*
- Never commit secrets. API keys live in `.env` (local) or `st.secrets` (deployment); `.env` is in `.gitignore`.
- The public demo must work **without an API key** (demo mode with pre-generated examples), so it never fails and never costs money.

## Tech stack
- Python 3.11+, Streamlit (multipage), pandas, numpy, plotly, yfinance
- LLM: Anthropic API (Claude) through a single wrapper in `core/llm.py`; the provider stays swappable
- Deployment: Streamlit Community Cloud
- Tests: pytest for `core/` functions

## Structure
```
app.py                     # entrypoint: st.navigation + landing page (problem, solution, modules, disclaimer)
app_pages/                 # not "pages/": that name makes Streamlit ignore st.navigation on deep links
  1_Risk_Monitor.py
  2_Meeting_Brief.py
  3_Market_Map.py
core/
  portfolios.py            # fictional client & portfolio generation (fixed seed)
  risk.py                  # risk metrics, limits, alerts, stress tests
  llm.py                   # LLM wrapper, guardrails, demo mode
  market_map.py            # LFN firm data loading, scoring, geocoding
data/                      # cached prices, demo outputs, firm dataset
tests/
.streamlit/config.toml     # theme
requirements.txt
README.md                  # English, with screenshots
```

## Modules
1. **Risk Monitor**: fictional client portfolios valued with real market data. Volatility, historical VaR, concentration, currency exposure, Lombard loan-to-value, risk-profile compliance, stress tests, automatic alerts.
2. **Meeting Brief (AI)**: generates a structured pre-meeting briefing for a fictional LFN client, with explicit guardrails and human review.
3. **LFN Market Map**: maps law, fiduciary and notary firms in Geneva and Vaud from public data and ranks prospects for business development.

## Coding standards
- Clean, typed (type hints), commented code; small pure functions in `core/`, UI only in `app_pages/` and `app.py`
- Every financial formula is documented in its docstring (definition, assumptions, limits)
- Fixed random seed, so the demo is reproducible
- Cache market data (`st.cache_data`) and fall back to a local CSV if yfinance fails
- English in code and UI

## Author's learning goal
The author must be able to explain every formula and design choice in an interview.
After each task, briefly explain **what** was built, **why**, and **which assumptions** were made.
