# LFN Advisor Copilot

*Student prototype — fictional data — not investment advice.*

**A prototype assistant for LFN client advisors: it monitors portfolio risk, prepares meetings with guardrailed AI,
and helps prioritise prospects.**

**Live demo:** [lfn-advisor-copilot.streamlit.app](https://lfn-advisor-copilot.streamlit.app) — no login, no API key needed.

Built by Yousif Bag — MSc Finance, HEC Lausanne · [GitHub](https://github.com/yxseef) ·
[LinkedIn](https://www.linkedin.com/in/yousif-bag-8002303b2/)

## In two minutes

In Swiss private banking, **LFN** teams serve **lawyers, fiduciaries and notaries**: their own wealth and firm
accounts, and the needs of the clients they represent (estates, property deals, escrow accounts). An advisor covers
dozens of such relationships and must spot risk early and prepare every meeting with limited time.

This independent student project shows three tools for that job. Every client and firm is invented, and the project
has no link to any bank.

| Module | What the advisor gets |
|---|---|
| **Risk Monitor** | Ten fictional portfolios valued every day with real market prices. Risk figures, profile limits and automatic alerts, ranked by severity. |
| **Meeting Brief (AI)** | A one-page pre-meeting draft for one client. The AI writes the narrative, the risk engine writes the figures, and the advisor must approve the draft before export. |
| **LFN Market Map** | Law, fiduciary and notary firms in Geneva and Vaud on a map, with a prospect score whose weights the advisor sets. The public demo uses fictional firms (see [Limits](#limits)). |

### Home

![Home: top navigation bar, project summary and the three modules](docs/screenshots/home.png)

### Risk Monitor

![Risk Monitor: book overview, one row per fictional client coloured by its most severe alert](docs/screenshots/risk_monitor.png)

### Meeting Brief

![Meeting Brief: AI-generated draft banner, figures from the risk engine, probable needs](docs/screenshots/meeting_brief.png)

### LFN Market Map

![Market Map: fictional firms placed on Geneva and Vaud commune centres, coloured by profession](docs/screenshots/market_map.png)

## Responsible AI

The Meeting Brief is the only module that calls a language model, and the guardrails are on by default:

1. **Only fictional data goes in.** The prompt contains one fictional client from the Risk Monitor. A free-text note
   with a name, an email, an IBAN or a Swiss AVS number is blocked before any call.
2. **Figures come from the risk engine.** AUM, volatility, VaR and alerts are written by the application. Any other
   number the model writes is marked "(to verify)".
3. **Discussion topics, not orders.** Banking solutions come from a closed list of generic product families. A sentence
   that reads like a trading instruction is rewritten as a discussion topic.
4. **A human signs off.** The draft carries an "AI-generated draft" banner, and PDF or Markdown export stays locked
   until the advisor clicks "Reviewed by advisor".
5. **No key, no cost.** Without an API key the page uses pre-written text and makes no call. With a key, live calls are
   capped at five per session, and the session journal records the time, prompt version, client and review status,
   never the brief, the note or the key.

## How this was built

I designed the project, wrote the specifications and validated the formulas. The code was written with an AI
assistant (Claude in Cursor), under my control.

---

## Details for technical readers

### Architecture

```
app.py               entrypoint: top navigation, shared style, Home page
assets/              text wordmark shown in the navigation bar
app_pages/           Streamlit pages (UI only)
  1_Risk_Monitor.py
  2_Meeting_Brief.py
  3_Market_Map.py
core/                pure, typed functions, no UI
  portfolios.py      fictional clients and portfolios (fixed seed)
  risk.py            prices, risk metrics, limits, stress tests, alerts
  llm.py             LLM wrapper, guardrails, demo mode, PDF/Markdown export
  market_map.py      firm cleaning, keyword classification, geocoding, prospect score
data/                price fallback, demo briefs, firm file, commune geocode cache
docs/screenshots/    images used here and on the Home page
tests/               pytest, no network
```

The pages folder is called `app_pages/`, not `pages/`. With a `pages/` folder, Streamlit opens a deep link such as
`/Market_Map` in its older folder mode and ignores the `st.navigation` menu.

### Risk methodology

- **Data.** Three years of daily adjusted closes from Yahoo Finance, converted to CHF with daily EUR/CHF and USD/CHF.
  Cached for one hour, with `data/prices_fallback.csv` if the download fails.
- **Volatility.** Standard deviation of daily portfolio returns × √252.
- **Historical VaR 95 %.** The loss not exceeded on 95 % of past days, applied to today's holdings. 10-day VaR = 1-day
  VaR × √10.
- **Expected Shortfall 95 %.** The average loss on the worst 5 % of days.
- **Risk contributions.** Euler decomposition, RCᵢ = wᵢ (Σw)ᵢ / σₚ. Contributions add up to total volatility.
- **Lombard loan.** Lending value = Σ market value × advance rate; usage = loan / lending value. Warning from 80 %,
  margin call above 100 %.
- **Stress tests.** Instant shocks: equities −20 %, EUR/CHF −10 % on EUR assets, +200 bp on bond ETFs
  (loss ≈ duration × 2 %).
- **Alerts.** Each risk profile (conservative, balanced, dynamic) has limits on equities, single equity, gold,
  foreign currency and volatility. Breach above a limit, warning from 90 % of it.

### Meeting Brief pipeline

One Anthropic call (`claude-sonnet-4-5` by default) through `core/llm.py`. The answer must be JSON that matches a
pydantic schema. Invalid JSON, a schema failure or an API error falls back to the pre-written brief and the page says
so. Pre-written text lives in `data/demo_briefs.json` and contains no digits, so figures are always today's.

### Market Map score

Score = 100 × weighted average of four criteria, each between 0 and 1. The weights are sliders on the page.

| Criterion | Points |
|---|---|
| Legal form | SA 1 · Sàrl 0.8 · SNC 0.5 · sole proprietorship or other 0.3 |
| Seniority | years since registration ÷ 30, capped at 1 |
| Profession | fiduciary 1 · notary 0.85 · lawyer 0.7 (also adjustable) |
| Local density | firms in the same commune ÷ firms in the busiest commune |

The score is a calling order for an advisor. It does not measure wealth, credit quality or the firm's clients, and
status (active, in liquidation) is shown but not scored.

### Limits

- **Fictional data everywhere.** Clients and portfolios are generated. Market prices are real but serve only as
  illustration.
- **Market Map uses an invented example.** The Zefix PublicREST API answers `401 Unauthorized` without credentials
  from the Federal Registry of Commerce. A search also needs a name of at least three characters, and the records
  carry no NOGA industry code. So the demo ranks invented firms, all named "Exemple…", placed on real communes.
  Filling `data/lfn_firms.csv` with a register extract replaces them.
- **Coverage and classification.** Sole practitioners who are not in the commercial register would be missing.
  Profession is classified from keywords (avocat, notaire, fiduciaire, Treuhand, révision), so a firm can be missed
  or mislabelled. "Étude" alone is ignored because it can mean a law office or a notary office.
- **Map points are commune centres** from swisstopo (geo.admin.ch), slightly offset, never street addresses.
- **Risk model.** Historical VaR assumes the next days look like the last three years. Stress shocks are
  instantaneous and applied one at a time.
- **AI checks are lexical.** The number check can accept a real figure used in the wrong sentence. The model can be
  wrong in a sentence with no figure. Review is a gate, not a guarantee.

### Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
pytest
```

Python 3.11 or later. The app runs without any key. For a live model, copy `.env.example` to `.env` (gitignored) and
set `ANTHROPIC_API_KEY`, or set it in Streamlit secrets. `ANTHROPIC_MODEL` is optional.

Refresh the price fallback with `python -m scripts.update_price_fallback`.

### Tech stack

Python · Streamlit · pandas · NumPy · Plotly · yfinance · Anthropic API (optional) · pydantic · fpdf2 ·
swisstopo geo.admin.ch · pytest · Streamlit Community Cloud.
