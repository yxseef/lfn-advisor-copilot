# LFN Advisor Copilot

*Student prototype — fictional data — not investment advice.*

An independent student project exploring how a client advisor serving **lawyers, fiduciaries and notaries (LFN)**
can use data and AI responsibly. No affiliation with any bank.

## Modules
- **Risk Monitor** (available): fictional portfolios valued with real market data — volatility, historical VaR and
  Expected Shortfall, concentration, currency exposure, Lombard loan-to-value, risk-profile compliance, stress tests
  and automatic alerts.
- **Meeting Brief (AI)** (available): a pre-meeting briefing for one fictional LFN client. Without an API key it uses pre-generated text plus live Risk Monitor figures. With a key, calls are capped at five per session. Guardrails: personal-data filter on the free-text note, schema-validated JSON, figures checked against the client data, no investment orders, advisor review before export, and a session journal.
- **LFN Market Map** (available): firms in Geneva and Vaud, a commune map and a transparent prospect score. Without a filled `data/lfn_firms.csv` the page shows an invented example (every name starts with "Exemple"). Zefix is not called: the public API requires credentials and does not return a NOGA code.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
pytest
```

Market data comes from Yahoo Finance and is cached locally. If the download fails, the app falls back to
`data/prices_fallback.csv` (refresh it with `python -m scripts.update_price_fallback`).

The Meeting Brief runs with no key. For a live model, set `ANTHROPIC_API_KEY` in a gitignored `.env`
file or in Streamlit secrets (`ANTHROPIC_MODEL` is optional). See `.env.example`.
