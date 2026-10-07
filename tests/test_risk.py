"""Unit tests for core/risk.py. Synthetic data only: no network access."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.portfolios import INSTRUMENTS, LombardLoan, all_tickers, build_clients
from core.risk import (
    FX_TICKERS,
    ComplianceCheck,
    Scenario,
    analyze_client,
    annualized_volatility,
    build_alerts,
    check_profile,
    clean_prices,
    concentration,
    daily_returns,
    expected_shortfall,
    get_all_alerts,
    historical_var,
    limit_status,
    load_prices,
    lombard_metrics,
    risk_contributions,
    run_stress_tests,
    scenario_pnl,
    to_chf,
    value_history,
)

SIGMA = 0.01


@pytest.fixture
def normal_returns() -> np.ndarray:
    return np.random.default_rng(0).normal(0.0, SIGMA, 200_000)


def make_positions(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    """rows: (ticker, asset_class, currency, value_chf, modified_duration)."""
    df = pd.DataFrame(rows, columns=["ticker", "asset_class", "currency", "value_chf", "modified_duration"])
    df = df.set_index("ticker")
    df["weight"] = df["value_chf"] / df["value_chf"].sum()
    return df


def synthetic_prices(tickers: list[str], days: int = 800, seed: int = 1) -> pd.DataFrame:
    """Geometric random walks for instruments and FX rates."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=days)
    cols = sorted(set(tickers) | set(FX_TICKERS.values()))
    rets = rng.normal(0.0002, 0.01, (days, len(cols)))
    start = np.array([0.95 if c == "EURCHF=X" else 0.88 if c == "USDCHF=X" else 100.0 for c in cols])
    return pd.DataFrame(start * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)


# --- Volatility, VaR, ES ----------------------------------------------------


def test_volatility_matches_sqrt_time_scaling(normal_returns: np.ndarray) -> None:
    assert annualized_volatility(normal_returns) == pytest.approx(SIGMA * math.sqrt(252), rel=0.01)


def test_volatility_of_constant_returns_is_zero() -> None:
    assert annualized_volatility(np.full(50, 0.001)) == pytest.approx(0.0)


def test_var_matches_normal_quantile(normal_returns: np.ndarray) -> None:
    assert historical_var(normal_returns, 0.95) == pytest.approx(1.6449 * SIGMA, rel=0.02)


def test_var_10d_is_sqrt10_times_1d(normal_returns: np.ndarray) -> None:
    assert historical_var(normal_returns, 0.95, 10) == pytest.approx(
        historical_var(normal_returns, 0.95, 1) * math.sqrt(10)
    )


def test_es_matches_normal_formula(normal_returns: np.ndarray) -> None:
    # For a normal distribution ES_95 = σ × φ(1.645) / 0.05 ≈ 2.063 σ
    assert expected_shortfall(normal_returns, 0.95) == pytest.approx(2.0627 * SIGMA, rel=0.02)


def test_es_is_at_least_var(normal_returns: np.ndarray) -> None:
    assert expected_shortfall(normal_returns) >= historical_var(normal_returns)


def test_es_hand_example() -> None:
    # One crash day among 20: the 5 % tail contains only that day.
    returns = np.array([-0.10] + [0.01] * 19)
    assert expected_shortfall(returns, 0.95) == pytest.approx(0.10)


def test_var_is_zero_when_no_losses() -> None:
    assert historical_var(np.linspace(0.001, 0.01, 100)) == 0.0


def test_metrics_need_two_observations() -> None:
    with pytest.raises(ValueError):
        historical_var([0.01])


# --- Valuation and FX --------------------------------------------------------


def test_to_chf_converts_with_fx_and_handles_cash() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame(
        {"SAP.DE": [100.0, 110.0, 120.0], "NESN.SW": [80.0, 81.0, 82.0], "EURCHF=X": [0.9, 0.95, 1.0],
         "USDCHF=X": [0.8, 0.8, 0.8]},
        index=idx,
    )
    chf = to_chf(prices, ["SAP.DE", "NESN.SW", "CASH_EUR", "CASH_CHF", "AAPL"])
    assert list(chf["SAP.DE"]) == pytest.approx([90.0, 104.5, 120.0])
    assert list(chf["NESN.SW"]) == pytest.approx([80.0, 81.0, 82.0])
    assert list(chf["CASH_EUR"]) == pytest.approx([0.9, 0.95, 1.0])
    assert list(chf["CASH_CHF"]) == pytest.approx([1.0, 1.0, 1.0])
    assert "AAPL" not in chf.columns  # no price data → skipped, not invented


def test_value_history_and_returns() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.1], "B": [1.0, 1.0, 1.0]}, index=idx)
    values = value_history({"A": 10, "B": 100}, prices)
    assert list(values) == pytest.approx([200.0, 210.0, 221.0])
    assert list(daily_returns(values)) == pytest.approx([0.05, 11 / 210])


def test_clean_prices_forward_fills_holidays() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [np.nan, 1.0, np.nan, 3.0], "B": [5.0, 5.0, 6.0, 7.0]}, index=idx)
    cleaned = clean_prices(prices)
    assert len(cleaned) == 3  # first row dropped (A not started yet)
    assert cleaned["A"].iloc[1] == 1.0  # holiday carried forward


# --- Risk contributions -------------------------------------------------------


def test_risk_contributions_sum_to_portfolio_vol() -> None:
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(0, [0.01, 0.02, 0.005], (5000, 3)), columns=["A", "B", "C"])
    rets["CASH"] = 0.0
    w = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "CASH": 0.1})
    rc = risk_contributions(rets, w)
    port_vol = math.sqrt(w.values @ (rets.cov().values * 252) @ w.values)
    assert rc["contribution"].sum() == pytest.approx(port_vol)
    assert rc["pct_of_risk"].sum() == pytest.approx(1.0)
    assert rc.loc["CASH", "contribution"] == pytest.approx(0.0)
    assert rc.index[0] == "B"  # most volatile asset dominates risk


# --- Concentration and compliance -------------------------------------------


def test_concentration_single_security_ignores_funds_and_cash() -> None:
    pos = make_positions([
        ("NESN.SW", "swiss_equity", "CHF", 150, 0),
        ("AAPL", "intl_equity", "USD", 100, 0),
        ("CSBGC3.SW", "bonds", "CHF", 600, 5),  # largest line, but a diversified fund
        ("CASH_EUR", "cash", "EUR", 150, 0),
    ])
    conc = concentration(pos)
    assert conc.max_security == "NESN.SW"
    assert conc.max_security_weight == pytest.approx(0.15)
    assert conc.equity_weight == pytest.approx(0.25)
    assert conc.non_chf_weight == pytest.approx(0.25)
    assert conc.by_asset_class["bonds"] == pytest.approx(0.60)


@pytest.mark.parametrize(
    ("value", "limit", "expected"),
    [(0.30, 0.35, "ok"), (0.315, 0.35, "warning"), (0.35, 0.35, "warning"), (0.36, 0.35, "breach")],
)
def test_limit_status_thresholds(value: float, limit: float, expected: str) -> None:
    assert limit_status(value, limit) == expected


def test_conservative_profile_equity_breach() -> None:
    pos = make_positions([
        ("NESN.SW", "swiss_equity", "CHF", 200, 0),
        ("NOVN.SW", "swiss_equity", "CHF", 200, 0),
        ("CASH_CHF", "cash", "CHF", 600, 0),
    ])
    checks = {c.rule.split(" (")[0]: c for c in check_profile(concentration(pos), 0.05, "conservative")}
    assert checks["Equity allocation"].status == "breach"  # 40 % > 35 %
    assert checks["Largest single equity"].status == "breach"  # 20 % > 10 %
    assert checks["Foreign-currency exposure"].status == "ok"
    assert checks["Annualised volatility"].status == "ok"
    # Same portfolio is fine for a dynamic profile, except the 20 % single line.
    dyn = {c.rule.split(" (")[0]: c for c in check_profile(concentration(pos), 0.05, "dynamic")}
    assert dyn["Equity allocation"].status == "ok"
    assert dyn["Largest single equity"].status == "breach"


# --- Lombard --------------------------------------------------------------------


def test_lombard_metrics() -> None:
    values = pd.Series({"swiss_equity": 1_000.0, "bonds": 1_000.0})
    loan = LombardLoan(1_000.0, {"swiss_equity": 0.70, "bonds": 0.85})
    lom = lombard_metrics(values, loan)
    assert lom.lending_value == pytest.approx(1_550.0)
    assert lom.ltv == pytest.approx(0.5)
    assert lom.usage == pytest.approx(1_000 / 1_550)
    assert lom.available_margin == pytest.approx(550.0)
    assert not lom.margin_call


def test_lombard_cushion_is_the_fall_that_triggers_margin_call() -> None:
    values = pd.Series({"swiss_equity": 1_000.0, "cash": 500.0})
    loan = LombardLoan(800.0, {"swiss_equity": 0.70, "cash": 0.95})
    lom = lombard_metrics(values, loan)
    after = lombard_metrics(values * (1 - lom.cushion), loan)
    assert after.usage == pytest.approx(1.0)
    assert lombard_metrics(values * (1 - lom.cushion - 0.01), loan).margin_call


def test_lombard_margin_call() -> None:
    lom = lombard_metrics(pd.Series({"intl_equity": 1_000.0}), LombardLoan(700.0, {"intl_equity": 0.60}))
    assert lom.margin_call
    assert lom.available_margin == pytest.approx(-100.0)


# --- Stress tests -----------------------------------------------------------------


@pytest.fixture
def mixed_positions() -> pd.DataFrame:
    return make_positions([
        ("NESN.SW", "swiss_equity", "CHF", 1_000, 0),
        ("SAP.DE", "intl_equity", "EUR", 1_000, 0),
        ("CSBGC3.SW", "bonds", "CHF", 1_000, 5.0),
        ("CASH_EUR", "cash", "EUR", 1_000, 0),
    ])


def test_equity_shock(mixed_positions: pd.DataFrame) -> None:
    pnl = scenario_pnl(mixed_positions, Scenario("eq", equity_shock=-0.20))
    assert pnl.sum() == pytest.approx(-400.0)
    assert pnl["CASH_EUR"] == 0.0


def test_fx_shock_hits_all_eur_positions(mixed_positions: pd.DataFrame) -> None:
    pnl = scenario_pnl(mixed_positions, Scenario("fx", fx_shocks={"EUR": -0.10}))
    assert pnl["SAP.DE"] == pytest.approx(-100.0)
    assert pnl["CASH_EUR"] == pytest.approx(-100.0)
    assert pnl["NESN.SW"] == 0.0


def test_rate_shock_uses_duration(mixed_positions: pd.DataFrame) -> None:
    pnl = scenario_pnl(mixed_positions, Scenario("rates", rate_shift_bp=200))
    assert pnl["CSBGC3.SW"] == pytest.approx(-100.0)  # -5 × 2 % × 1000
    assert pnl.drop("CSBGC3.SW").abs().sum() == 0.0


def test_combined_shock_compounds(mixed_positions: pd.DataFrame) -> None:
    pnl = scenario_pnl(mixed_positions, Scenario("both", equity_shock=-0.20, fx_shocks={"EUR": -0.10}))
    assert pnl["SAP.DE"] == pytest.approx(-280.0)  # (0.8 × 0.9 - 1) × 1000


def test_stress_flags_margin_call() -> None:
    pos = make_positions([("NESN.SW", "swiss_equity", "CHF", 1_000, 0)])
    loan = LombardLoan(600.0, {"swiss_equity": 0.70})  # usage 86 % today
    stress = run_stress_tests(pos, loan, [Scenario("eq", equity_shock=-0.20)])
    assert stress.loc["eq", "lombard_usage_after"] == pytest.approx(600 / 560)
    assert bool(stress.loc["eq", "margin_call"])


# --- Alerts -----------------------------------------------------------------------


def test_alerts_sorted_by_severity_and_margin_call_is_breach() -> None:
    checks = [
        ComplianceCheck("Equity allocation", 0.33, 0.35, "warning"),
        ComplianceCheck("Gold allocation", 0.02, 0.10, "ok"),
    ]
    lom = lombard_metrics(pd.Series({"intl_equity": 1_000.0}), LombardLoan(700.0, {"intl_equity": 0.60}))
    alerts = build_alerts(checks, lom, pd.DataFrame(), missing_tickers=["XYZ"], profile="conservative")
    assert [a.severity for a in alerts] == ["breach", "warning", "warning"]
    assert alerts[0].category == "Lombard"
    assert {a.category for a in alerts} == {"Lombard", "Concentration", "Data"}


def test_comfortable_lombard_is_info_only() -> None:
    lom = lombard_metrics(pd.Series({"bonds": 1_000.0}), LombardLoan(100.0, {"bonds": 0.85}))
    alerts = build_alerts([], lom, pd.DataFrame())
    assert [a.severity for a in alerts] == ["info"]


# --- Data loading -------------------------------------------------------------------


def _offline(*_: object) -> pd.DataFrame:
    raise ConnectionError("no network in tests")


def test_load_prices_falls_back_to_csv(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback.csv"
    synthetic_prices(["NESN.SW"]).to_csv(fallback)
    data = load_prices(["NESN.SW", "AAPL"], cache_path=tmp_path / "cache.csv",
                       fallback_path=fallback, downloader=_offline)
    assert data.source == "fallback CSV"
    assert data.missing == ["AAPL"]
    assert "NESN.SW" in data.prices.columns


def test_load_prices_writes_and_reuses_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache.csv"
    calls: list[int] = []

    def fake_download(tickers, years):  # type: ignore[no-untyped-def]
        calls.append(1)
        return synthetic_prices(list(tickers))

    first = load_prices(["NESN.SW"], cache_path=cache, fallback_path=tmp_path / "none.csv",
                        downloader=fake_download)
    second = load_prices(["NESN.SW"], cache_path=cache, fallback_path=tmp_path / "none.csv",
                         downloader=fake_download)
    assert first.source.startswith("Yahoo")
    assert second.source.startswith("local cache")
    assert len(calls) == 1


def test_load_prices_raises_without_any_source(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        load_prices(["NESN.SW"], cache_path=tmp_path / "a.csv", fallback_path=tmp_path / "b.csv",
                    downloader=_offline)


# --- End to end ------------------------------------------------------------------------


def test_analyze_client_end_to_end() -> None:
    clients = build_clients()
    prices = synthetic_prices(all_tickers(clients))
    results = [analyze_client(c, prices) for c in clients]

    for r in results:
        assert r.positions["weight"].sum() == pytest.approx(1.0)
        assert r.aum == pytest.approx(r.history.iloc[-1])
        assert 0 < r.var_1d <= r.es_1d
        assert r.var_10d == pytest.approx(r.var_1d * math.sqrt(10))
        assert set(r.stress.index) == {"Equities -20%", "EUR/CHF -10%", "Rates +200 bp"}
        assert (r.lombard is not None) == (r.client.lombard is not None)

    alerts = get_all_alerts(results)
    assert set(alerts["severity"]) <= {"info", "warning", "breach"}
    # Deliberate issues in the fictional book must be detected.
    by_id = {r.client.client_id: r for r in results}
    assert any("NESN.SW" in a.message and a.severity == "breach" for a in by_id["C03"].alerts)
    assert any("UBSG.SW" in a.message and a.severity == "breach" for a in by_id["C07"].alerts)


def test_every_held_instrument_is_known() -> None:
    for client in build_clients():
        assert all(t in INSTRUMENTS for t in client.tickers)
