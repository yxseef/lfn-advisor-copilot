"""Risk engine: market data, risk metrics, limits, stress tests and alerts.

All functions are pure (no Streamlit) so they can be unit-tested. Caching for
the UI is done in the page with ``st.cache_data``; this module adds a disk
cache plus a CSV fallback so the demo works even if Yahoo Finance is down.

Conventions
-----------
* All amounts are in CHF (the clients' reference currency).
* Returns are simple daily returns: r_t = V_t / V_{t-1} - 1.
* Risk measures (VaR, ES) are reported as *positive* numbers = losses.
* Weights are fractions (0.25 = 25 %).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from core.portfolios import (
    EQUITY_CLASSES,
    INSTRUMENTS,
    REFERENCE_CURRENCY,
    Client,
    LombardLoan,
    RiskProfile,
)

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
HISTORY_YEARS = 3
FX_TICKERS: dict[str, str] = {"EUR": "EURCHF=X", "USD": "USDCHF=X"}  # quoted as CHF per 1 unit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / "prices_cache.csv"
FALLBACK_PATH = DATA_DIR / "prices_fallback.csv"

Severity = Literal["info", "warning", "breach"]
Status = Literal["ok", "warning", "breach"]
SEVERITY_RANK: dict[str, int] = {"ok": 0, "info": 1, "warning": 2, "breach": 3}


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceData:
    """Local-currency closing prices (instruments + FX rates).

    Attributes:
        prices: Date-indexed DataFrame, one column per ticker.
        source: Human-readable description of where the data came from.
        missing: Requested tickers for which no data was found anywhere.
    """

    prices: pd.DataFrame
    source: str
    missing: list[str] = field(default_factory=list)


def download_prices(tickers: Sequence[str], years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Download adjusted daily closes from Yahoo Finance.

    ``auto_adjust=True`` returns prices adjusted for splits and dividends, so
    returns are total returns (dividends reinvested), which is the right basis
    for risk measurement.
    """
    import yfinance as yf  # imported lazily so tests never need the network

    raw = yf.download(list(tickers), period=f"{years}y", auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(axis=1, how="all")


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    except OSError as exc:  # read-only file system on some hosts
        logger.warning("Could not write %s: %s", path, exc)


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def clean_prices(prices: pd.DataFrame, years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Align series from different exchanges on one calendar.

    Swiss, US and euro-area markets have different holidays. Missing closes
    are forward-filled (max 5 days), i.e. the last known price is carried
    forward, which produces a zero return on local holidays. Dates before all
    series have started are dropped.
    """
    df = prices.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if not df.empty:
        start = df.index.max() - pd.DateOffset(years=years)
        df = df[df.index >= start]
    return df.ffill(limit=5).dropna(how="any")


def load_prices(
    tickers: Iterable[str],
    years: int = HISTORY_YEARS,
    cache_path: Path = CACHE_PATH,
    fallback_path: Path = FALLBACK_PATH,
    max_age_hours: float = 24.0,
    downloader: Callable[[Sequence[str], int], pd.DataFrame] = download_prices,
) -> PriceData:
    """Load prices for ``tickers`` plus the FX rates needed for CHF conversion.

    Order of preference:
      1. Fresh disk cache (< ``max_age_hours``) containing every ticker.
      2. Live download from Yahoo Finance (then written to the cache).
      3. Stale disk cache.
      4. Committed fallback CSV.
    Any ticker still missing after step 2/3 is filled from the fallback CSV
    when possible. Tickers found nowhere are reported in ``missing``.
    """
    needed = sorted(set(tickers) | set(FX_TICKERS.values()))
    cached = _read_csv(cache_path)
    fallback = _read_csv(fallback_path)

    if cached is not None and _is_fresh(cache_path, max_age_hours) and set(needed) <= set(cached.columns):
        return PriceData(clean_prices(cached[needed], years), "local cache (Yahoo Finance)", [])

    try:
        live = downloader(needed, years)
    except Exception as exc:  # network errors, rate limits, API changes…
        logger.warning("Price download failed: %s", exc)
        live = pd.DataFrame()

    if not live.empty:
        base, source = live, "Yahoo Finance (live)"
        _write_csv(live, cache_path)
    elif cached is not None:
        base, source = cached, "stale local cache"
    elif fallback is not None:
        base, source = fallback, "fallback CSV"
    else:
        raise RuntimeError("No price data available: download failed and no local CSV found.")

    gaps = [t for t in needed if t not in base.columns]
    if gaps and fallback is not None and base is not fallback:
        fill = [t for t in gaps if t in fallback.columns]
        if fill:
            base = base.join(fallback[fill], how="outer")
            source += " + fallback CSV"

    available = [t for t in needed if t in base.columns]
    missing = [t for t in needed if t not in base.columns]
    return PriceData(clean_prices(base[available], years), source, missing)


def to_chf(prices: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Convert local-currency prices into CHF.

    Formula: P_CHF(t) = P_local(t) × FX(t), with FX quoted as CHF per unit of
    foreign currency (e.g. EURCHF = 0.93). Cash accounts have P_local = 1, so
    their CHF price is simply the FX rate (1 for CHF cash).

    Tickers without price data (or without the needed FX rate) are skipped.
    """
    out: dict[str, pd.Series] = {}
    ones = pd.Series(1.0, index=prices.index)
    for ticker in tickers:
        inst = INSTRUMENTS[ticker]
        if inst.currency == REFERENCE_CURRENCY:
            fx = ones
        elif FX_TICKERS.get(inst.currency) in prices.columns:
            fx = prices[FX_TICKERS[inst.currency]]
        else:
            continue
        if inst.asset_class == "cash":
            out[ticker] = fx.copy()
        elif ticker in prices.columns:
            out[ticker] = prices[ticker] * fx
    return pd.DataFrame(out, index=prices.index)


# ---------------------------------------------------------------------------
# Portfolio valuation
# ---------------------------------------------------------------------------


def position_table(client: Client, prices_chf: pd.DataFrame) -> pd.DataFrame:
    """Current positions valued at the latest available CHF price.

    Market value: V_i = q_i × P_i,CHF ;  weight: w_i = V_i / Σ_j V_j.
    """
    last = prices_chf.iloc[-1]
    rows = []
    for ticker, qty in client.quantities.items():
        if ticker not in prices_chf.columns:
            continue
        inst = INSTRUMENTS[ticker]
        rows.append(
            {
                "ticker": ticker,
                "name": inst.name,
                "asset_class": inst.asset_class,
                "currency": inst.currency,
                "quantity": qty,
                "price_chf": float(last[ticker]),
                "value_chf": qty * float(last[ticker]),
                "modified_duration": inst.modified_duration,
            }
        )
    df = pd.DataFrame(rows).set_index("ticker")
    df["weight"] = df["value_chf"] / df["value_chf"].sum()
    return df.sort_values("value_chf", ascending=False)


def value_history(quantities: dict[str, float], prices_chf: pd.DataFrame) -> pd.Series:
    """Hypothetical CHF value of the *current* holdings over the history.

    V_t = Σ_i q_i × P_i,CHF(t), with quantities held constant (buy-and-hold).
    This is the standard basis for historical-simulation VaR: "what would
    today's portfolio have done on each past day?". It is not the client's
    actual past performance (no trades, fees or flows are modelled).
    """
    cols = [t for t in quantities if t in prices_chf.columns]
    q = pd.Series({t: quantities[t] for t in cols})
    return (prices_chf[cols] * q).sum(axis=1).rename("value_chf")


def daily_returns(values: pd.Series) -> pd.Series:
    """Simple daily returns r_t = V_t / V_{t-1} - 1."""
    return values.pct_change().dropna()


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


def _as_array(returns: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        raise ValueError("At least two return observations are required.")
    return arr


def annualized_volatility(returns: pd.Series | np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized volatility.

    Definition: σ_annual = σ_daily × √252, where σ_daily is the sample
    standard deviation (ddof=1) of daily returns.
    Assumption: returns are i.i.d., so variance grows linearly with time.
    Limits: treats upside and downside moves symmetrically; underestimates
    risk when volatility clusters (calm past ≠ calm future).
    """
    arr = _as_array(returns)
    return float(np.std(arr, ddof=1) * math.sqrt(periods_per_year))


def historical_var(
    returns: pd.Series | np.ndarray, confidence: float = 0.95, horizon_days: int = 1
) -> float:
    """Historical Value-at-Risk, as a positive fraction of portfolio value.

    Definition: VaR_α = -q_{1-α}(r), the (1-α) empirical quantile of daily
    returns, sign-flipped. At 95 %, "on 19 days out of 20 the loss should not
    exceed VaR".
    Horizon: VaR_h = VaR_1d × √h (square-root-of-time rule, as in Basel).
    Assumptions: the past window (3 years) is representative of tomorrow;
    returns are i.i.d. for the √h scaling.
    Limits: says nothing about the size of losses beyond the quantile; the
    √h rule ignores autocorrelation and fat-tail compounding.
    """
    arr = _as_array(returns)
    q = float(np.quantile(arr, 1.0 - confidence))
    return max(-q, 0.0) * math.sqrt(horizon_days)


def expected_shortfall(
    returns: pd.Series | np.ndarray, confidence: float = 0.95, horizon_days: int = 1
) -> float:
    """Historical Expected Shortfall (CVaR), as a positive fraction.

    Definition: ES_α = -E[r | r ≤ q_{1-α}], the average return on the worst
    (1-α) share of days. Always ≥ VaR at the same confidence level.
    Why: unlike VaR it measures *how bad* the tail is, and it is a coherent
    risk measure (sub-additive: diversification never increases it).
    Horizon scaling with √h, same i.i.d. assumption as VaR. With 3 years of
    data the 5 % tail holds only ~38 observations, so ES is noisy.
    """
    arr = _as_array(returns)
    q = np.quantile(arr, 1.0 - confidence)
    tail = arr[arr <= q]
    return max(-float(tail.mean()), 0.0) * math.sqrt(horizon_days)


def risk_contributions(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Decompose portfolio volatility into position contributions (Euler).

    With Σ the annualized covariance matrix of asset returns (CHF):
      σ_p  = √(wᵀ Σ w)
      RC_i = w_i × (Σ w)_i / σ_p        (contribution, in volatility units)
      %RC_i = RC_i / σ_p                (share of total risk, sums to 100 %)
    A position can contribute more (or less) risk than its weight; a negative
    contribution means it hedges the rest of the portfolio.
    Assumption: constant weights (daily rebalanced), so σ_p can differ
    slightly from the buy-and-hold volatility.
    """
    cols = [c for c in weights.index if c in asset_returns.columns]
    w = weights[cols].to_numpy(dtype=float)
    cov = asset_returns[cols].cov().to_numpy() * TRADING_DAYS
    port_vol = float(np.sqrt(w @ cov @ w))
    if port_vol == 0:
        contrib = np.zeros_like(w)
    else:
        contrib = w * (cov @ w) / port_vol
    pct = contrib / port_vol if port_vol else contrib
    return pd.DataFrame(
        {"weight": w, "contribution": contrib, "pct_of_risk": pct}, index=cols
    ).sort_values("pct_of_risk", ascending=False)


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Concentration:
    """Portfolio concentration figures (all weights in % of total value)."""

    max_security: str | None
    max_security_weight: float
    by_asset_class: pd.Series
    by_currency: pd.Series
    equity_weight: float
    non_chf_weight: float


def concentration(positions: pd.DataFrame) -> Concentration:
    """Weights by single security, asset class and currency.

    "Single security" covers direct equity lines only: bond ETFs are already
    diversified funds, gold is governed by its own asset-class limit, and cash
    is not an issuer risk in this model.
    Currency exposure is by quotation currency (e.g. GLD counts as USD).
    """
    by_class = positions.groupby("asset_class")["weight"].sum().sort_values(ascending=False)
    by_ccy = positions.groupby("currency")["weight"].sum().sort_values(ascending=False)
    equities = positions[positions["asset_class"].isin(EQUITY_CLASSES)]
    if equities.empty:
        top, top_w = None, 0.0
    else:
        top = str(equities["weight"].idxmax())
        top_w = float(equities["weight"].max())
    return Concentration(
        max_security=top,
        max_security_weight=top_w,
        by_asset_class=by_class,
        by_currency=by_ccy,
        equity_weight=float(by_class.reindex(list(EQUITY_CLASSES)).fillna(0).sum()),
        non_chf_weight=float(1.0 - by_ccy.get(REFERENCE_CURRENCY, 0.0)),
    )


# ---------------------------------------------------------------------------
# Lombard credit
# ---------------------------------------------------------------------------

LOMBARD_WARNING_USAGE = 0.80  # warn when 80 % of the lending value is used


@dataclass(frozen=True)
class LombardResult:
    """Lombard loan metrics.

    Attributes:
        loan: Drawn amount (CHF).
        market_value: Market value of the pledged portfolio (CHF).
        lending_value: Collateral value after advance rates (CHF).
        ltv: Loan-to-value = loan / market value.
        usage: Loan / lending value (≥ 100 % means margin call).
        available_margin: Lending value - loan (negative = shortfall).
        cushion: Uniform market fall the portfolio can absorb before a margin
            call, = 1 - usage.
        margin_call: True when the loan exceeds the lending value.
    """

    loan: float
    market_value: float
    lending_value: float
    ltv: float
    usage: float
    available_margin: float
    cushion: float
    margin_call: bool


def lombard_metrics(values_by_class: pd.Series, loan: LombardLoan) -> LombardResult:
    """Compute collateral and loan-to-value figures for a Lombard loan.

    Formulas (values in CHF, per asset class c):
      Lending value  LV = Σ_c MV_c × advance_rate_c
      LTV            = Loan / Σ_c MV_c
      Usage          = Loan / LV
      Free margin    = LV - Loan
      Cushion        = 1 - Usage   (if all assets fall by x %, LV falls by x %)
      Margin call    ⇔ Loan > LV
    Assumptions: the whole portfolio is pledged; the loan is in CHF so it
    carries no FX risk; advance rates are applied per asset class (a real bank
    would also haircut concentrated or illiquid lines).
    """
    mv = float(values_by_class.sum())
    lv = float(sum(v * loan.advance_rates.get(cls, 0.0) for cls, v in values_by_class.items()))
    usage = loan.amount_chf / lv if lv > 0 else math.inf
    return LombardResult(
        loan=loan.amount_chf,
        market_value=mv,
        lending_value=lv,
        ltv=loan.amount_chf / mv if mv > 0 else math.inf,
        usage=usage,
        available_margin=lv - loan.amount_chf,
        cushion=1.0 - usage,
        margin_call=loan.amount_chf > lv,
    )


# ---------------------------------------------------------------------------
# Risk-profile compliance
# ---------------------------------------------------------------------------

WARNING_RATIO = 0.90  # a metric at ≥ 90 % of its limit triggers a warning


@dataclass(frozen=True)
class ProfileLimits:
    """Investment limits for a risk profile (fractions of portfolio value)."""

    max_equity: float
    max_single_security: float
    max_gold: float
    max_non_chf: float
    max_volatility: float


PROFILE_LIMITS: dict[str, ProfileLimits] = {
    "conservative": ProfileLimits(0.35, 0.10, 0.10, 0.30, 0.08),
    "balanced": ProfileLimits(0.60, 0.10, 0.10, 0.50, 0.12),
    "dynamic": ProfileLimits(0.85, 0.15, 0.15, 0.70, 0.18),
}


@dataclass(frozen=True)
class ComplianceCheck:
    rule: str
    value: float
    limit: float
    status: Status


def limit_status(value: float, limit: float, warning_ratio: float = WARNING_RATIO) -> Status:
    """``breach`` above the limit, ``warning`` from ``warning_ratio × limit``, else ``ok``."""
    if value > limit:
        return "breach"
    if value >= warning_ratio * limit:
        return "warning"
    return "ok"


def check_profile(conc: Concentration, volatility: float, profile: RiskProfile) -> list[ComplianceCheck]:
    """Compare portfolio figures to the limits of the client's risk profile."""
    lim = PROFILE_LIMITS[profile]
    gold = float(conc.by_asset_class.get("gold", 0.0))
    single = f"Largest single equity ({conc.max_security})" if conc.max_security else "Largest single equity"
    checks = [
        ("Equity allocation", conc.equity_weight, lim.max_equity),
        (single, conc.max_security_weight, lim.max_single_security),
        ("Gold allocation", gold, lim.max_gold),
        ("Foreign-currency exposure", conc.non_chf_weight, lim.max_non_chf),
        ("Annualized volatility", volatility, lim.max_volatility),
    ]
    return [ComplianceCheck(rule, value, limit, limit_status(value, limit)) for rule, value, limit in checks]


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """An instantaneous shock applied to current positions.

    Attributes:
        name: Display name.
        equity_shock: Relative price change of all equities (local currency).
        fx_shocks: Relative change of CHF value per currency (e.g. EUR: -0.10).
        rate_shift_bp: Parallel yield-curve shift in basis points.
    """

    name: str
    equity_shock: float = 0.0
    fx_shocks: dict[str, float] = field(default_factory=dict)
    rate_shift_bp: float = 0.0


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Equities -20%", equity_shock=-0.20),
    Scenario("EUR/CHF -10%", fx_shocks={"EUR": -0.10}),
    Scenario("Rates +200 bp", rate_shift_bp=200),
)


def scenario_pnl(positions: pd.DataFrame, scenario: Scenario) -> pd.Series:
    """Profit and loss (CHF) per position under a scenario.

    Per position i with CHF value V_i:
      Equity shock:  ΔV_i = V_i × s_eq                     (equities only)
      Rate shock:    ΔV_i ≈ -D_i × Δy × V_i                 (bond ETFs, D = modified duration)
      FX shock:      ΔV_i = (V_i + ΔV_i^local) × s_fx       (positions in that currency)
    The FX shock applies to the value after the local-market shock, so
    combined scenarios compound correctly: (1 + s_local)(1 + s_fx) - 1.
    Assumptions: instantaneous shock, no convexity (duration only, which
    overstates losses slightly for a +200 bp move), parallel shift on all
    curves, no second-order effects (e.g. equities reacting to FX).
    """
    v = positions["value_chf"]
    local = pd.Series(0.0, index=positions.index)
    is_equity = positions["asset_class"].isin(EQUITY_CLASSES)
    local[is_equity] = v[is_equity] * scenario.equity_shock
    is_bond = positions["asset_class"] == "bonds"
    local[is_bond] = -positions.loc[is_bond, "modified_duration"] * scenario.rate_shift_bp / 10_000 * v[is_bond]

    fx = positions["currency"].map(scenario.fx_shocks).fillna(0.0)
    return local + (v + local) * fx


def run_stress_tests(
    positions: pd.DataFrame,
    loan: LombardLoan | None = None,
    scenarios: Sequence[Scenario] = SCENARIOS,
) -> pd.DataFrame:
    """Run every scenario; report P&L and, if relevant, post-stress Lombard usage."""
    total = float(positions["value_chf"].sum())
    rows = []
    for sc in scenarios:
        pnl = scenario_pnl(positions, sc)
        row: dict[str, object] = {
            "scenario": sc.name,
            "pnl_chf": float(pnl.sum()),
            "pnl_pct": float(pnl.sum()) / total,
            "value_after_chf": total + float(pnl.sum()),
        }
        if loan is not None:
            stressed = (positions["value_chf"] + pnl).groupby(positions["asset_class"]).sum()
            lom = lombard_metrics(stressed, loan)
            row["lombard_usage_after"] = lom.usage
            row["margin_call"] = lom.margin_call
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alert:
    severity: Severity
    category: str
    message: str


def _pct(x: float) -> str:
    return f"{x:.1%}"


def build_alerts(
    checks: Sequence[ComplianceCheck],
    lombard: LombardResult | None,
    stress: pd.DataFrame,
    missing_tickers: Sequence[str] = (),
    profile: str = "",
) -> list[Alert]:
    """Turn metrics into prioritized alerts.

    Severity scale:
      * breach  – a limit is exceeded or a margin call is due: act now.
      * warning – within 10 % of a limit, Lombard usage ≥ 80 %, a stress
                  scenario would trigger a margin call, or data is missing.
      * info    – useful context (e.g. Lombard loan in place, comfortable).
    """
    alerts: list[Alert] = []

    for c in checks:
        category = "Volatility" if "volatility" in c.rule.lower() else (
            "Currency" if "currency" in c.rule.lower() else "Concentration"
        )
        if c.status == "breach":
            alerts.append(Alert("breach", category,
                                f"{c.rule} at {_pct(c.value)} exceeds the {profile} limit of {_pct(c.limit)}."))
        elif c.status == "warning":
            alerts.append(Alert("warning", category,
                                f"{c.rule} at {_pct(c.value)} is close to the {profile} limit of {_pct(c.limit)}."))

    if lombard is not None:
        if lombard.margin_call:
            alerts.append(Alert("breach", "Lombard",
                                f"Margin call: loan exceeds lending value by CHF {-lombard.available_margin:,.0f} "
                                f"(usage {_pct(lombard.usage)})."))
        elif lombard.usage >= LOMBARD_WARNING_USAGE:
            alerts.append(Alert("warning", "Lombard",
                                f"Lombard usage at {_pct(lombard.usage)}: a {_pct(lombard.cushion)} market fall "
                                f"would trigger a margin call."))
        else:
            alerts.append(Alert("info", "Lombard",
                                f"Lombard loan of CHF {lombard.loan:,.0f}, usage {_pct(lombard.usage)}, "
                                f"free margin CHF {lombard.available_margin:,.0f}."))

        if not lombard.margin_call and "margin_call" in stress.columns:
            for name in stress.index[stress["margin_call"].astype(bool)]:
                alerts.append(Alert("warning", "Stress",
                                    f"Scenario '{name}' would trigger a margin call "
                                    f"(usage {_pct(stress.loc[name, 'lombard_usage_after'])})."))

    for t in missing_tickers:
        alerts.append(Alert("warning", "Data", f"No price data for {t}: position excluded from all figures."))

    return sorted(alerts, key=lambda a: SEVERITY_RANK[a.severity], reverse=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientRisk:
    """Full risk picture of one client."""

    client: Client
    positions: pd.DataFrame
    history: pd.Series
    aum: float
    volatility: float
    var_1d: float
    var_10d: float
    es_1d: float
    es_10d: float
    concentration: Concentration
    contributions: pd.DataFrame
    compliance: list[ComplianceCheck]
    lombard: LombardResult | None
    stress: pd.DataFrame
    alerts: list[Alert]

    @property
    def status(self) -> Status:
        """Worst alert severity, with ``info`` treated as ``ok``."""
        worst = max((SEVERITY_RANK[a.severity] for a in self.alerts), default=0)
        return {3: "breach", 2: "warning"}.get(worst, "ok")  # type: ignore[return-value]


def analyze_client(client: Client, prices_local: pd.DataFrame, confidence: float = 0.95) -> ClientRisk:
    """Compute every risk figure and alert for one client.

    VaR and ES are converted to CHF by multiplying the return-based fraction
    by the current portfolio value.
    """
    prices_chf = to_chf(prices_local, client.tickers)
    missing = [t for t in client.tickers if t not in prices_chf.columns]

    positions = position_table(client, prices_chf)
    history = value_history(client.quantities, prices_chf)
    rets = daily_returns(history)
    aum = float(positions["value_chf"].sum())
    vol = annualized_volatility(rets)

    asset_rets = prices_chf[positions.index].pct_change().dropna()
    contrib = risk_contributions(asset_rets, positions["weight"])
    contrib.insert(0, "name", positions.loc[contrib.index, "name"])

    conc = concentration(positions)
    checks = check_profile(conc, vol, client.risk_profile)
    lombard = None
    if client.lombard is not None:
        lombard = lombard_metrics(positions.groupby("asset_class")["value_chf"].sum(), client.lombard)
    stress = run_stress_tests(positions, client.lombard)

    return ClientRisk(
        client=client,
        positions=positions,
        history=history,
        aum=aum,
        volatility=vol,
        var_1d=historical_var(rets, confidence, 1) * aum,
        var_10d=historical_var(rets, confidence, 10) * aum,
        es_1d=expected_shortfall(rets, confidence, 1) * aum,
        es_10d=expected_shortfall(rets, confidence, 10) * aum,
        concentration=conc,
        contributions=contrib,
        compliance=checks,
        lombard=lombard,
        stress=stress,
        alerts=build_alerts(checks, lombard, stress, missing, client.risk_profile),
    )


def get_all_alerts(results: Sequence[ClientRisk]) -> pd.DataFrame:
    """Every alert of the book, one row per alert, most severe first."""
    rows = [
        {"client_id": r.client.client_id, "client": r.client.name,
         "severity": a.severity, "category": a.category, "message": a.message}
        for r in results for a in r.alerts
    ]
    df = pd.DataFrame(rows, columns=["client_id", "client", "severity", "category", "message"])
    if df.empty:
        return df
    df["_rank"] = df["severity"].map(SEVERITY_RANK)
    return df.sort_values(["_rank", "client_id"], ascending=[False, True]).drop(columns="_rank").reset_index(drop=True)


def overview_table(results: Sequence[ClientRisk]) -> pd.DataFrame:
    """One row per client for the book-level dashboard."""
    rows = []
    for r in results:
        counts = {s: sum(a.severity == s for a in r.alerts) for s in ("breach", "warning", "info")}
        rows.append(
            {
                "Client": r.client.name,
                "Type": r.client.client_type.capitalize(),
                "Account": r.client.account_type.capitalize(),
                "Profile": r.client.risk_profile.capitalize(),
                "AUM (CHF)": r.aum,
                "Volatility": r.volatility,
                "VaR 95% 1d (CHF)": r.var_1d,
                "VaR 1d (% AUM)": r.var_1d / r.aum,
                "LTV": r.lombard.ltv if r.lombard else np.nan,
                "Breaches": counts["breach"],
                "Warnings": counts["warning"],
                "Alerts": counts["breach"] + counts["warning"] + counts["info"],
                "Status": r.status,
            }
        )
    return pd.DataFrame(rows)
