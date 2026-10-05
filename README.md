# LFN Advisor Copilot

*Student prototype — fictional data — not investment advice.*

An independent student project exploring how a client advisor serving **lawyers, fiduciaries and notaries (LFN)**
can use data and AI responsibly. No affiliation with any bank.

## Modules
- **Risk Monitor** (available): fictional portfolios valued with real market data — volatility, historical VaR and
  Expected Shortfall, concentration, currency exposure, Lombard loan-to-value, risk-profile compliance, stress tests
  and automatic alerts.
- **Meeting Brief (AI)** (planned)
- **LFN Market Map** (planned)

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
pytest
```

Market data comes from Yahoo Finance and is cached locally. If the download fails, the app falls back to
`data/prices_fallback.csv` (refresh it with `python -m scripts.update_price_fallback`).
