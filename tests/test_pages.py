"""Smoke tests: every page renders for every client without raising (offline)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import risk

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def offline_prices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Force the CSV fallback so tests never touch the network or the real cache."""
    def offline(*_: object) -> pd.DataFrame:
        raise ConnectionError("offline")

    monkeypatch.setattr(
        risk, "load_prices", partial(risk.load_prices, cache_path=tmp_path / "cache.csv", downloader=offline)
    )


def test_landing_page_renders() -> None:
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not at.exception
    assert any("not investment advice" in w.value for w in at.warning)


def test_risk_monitor_renders_for_every_client() -> None:
    at = AppTest.from_file(str(ROOT / "pages" / "1_Risk_Monitor.py"), default_timeout=60).run()
    assert not at.exception
    assert any("not investment advice" in w.value for w in at.warning)
    select = at.selectbox[0]
    for name in select.options:
        select.set_value(name.split("  ·  ")[0]).run()
        assert not at.exception, name
