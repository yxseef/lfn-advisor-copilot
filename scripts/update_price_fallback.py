"""Refresh data/prices_fallback.csv, the offline backup used when Yahoo Finance fails.

Usage (from the project root):  python -m scripts.update_price_fallback
"""

from __future__ import annotations

from core.portfolios import all_tickers, build_clients
from core.risk import FALLBACK_PATH, FX_TICKERS, HISTORY_YEARS, download_prices


def main() -> None:
    tickers = sorted(set(all_tickers(build_clients())) | set(FX_TICKERS.values()))
    prices = download_prices(tickers, HISTORY_YEARS)
    missing = sorted(set(tickers) - set(prices.columns))
    FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.round(6).to_csv(FALLBACK_PATH)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} tickers to {FALLBACK_PATH}")
    if missing:
        print(f"Missing tickers: {missing}")


if __name__ == "__main__":
    main()
