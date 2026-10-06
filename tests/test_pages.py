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
    assert any("Available" in m.value and "Meeting Brief" in m.value for m in at.markdown)
    assert any("Available" in m.value and "Market Map" in m.value for m in at.markdown)
    assert any("guardrailed AI" in m.value for m in at.markdown)
    assert any("Zefix" in i.value for i in at.info)
    assert any("Ask the Book" in m.value and "read-only" in m.value for m in at.markdown)
    assert len(at.get("image")) == 4
    assert any("Yousif Bag" in m.value for m in at.markdown)
    assert any(s.value == "Why I built this" for s in at.subheader)
    assert not any(s.value in {"The problem", "The solution"} for s in at.subheader)


def test_floating_ask_window_answers_on_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.llm.resolve_api_key", lambda: None)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    at.chat_input(key="ask_fab_input").set_value("Which conservative clients breach a limit?").run()
    assert not at.exception, at.exception
    assert any("AI-generated" in m.value and "Demo mode" in m.value for m in at.markdown)
    assert any(e.label.startswith("Tools used (2)") for e in at.expander)
    at.button(key="fab_example_3").click().run()
    assert any("read-only" in m.value for m in at.markdown)


def test_ask_the_book_page_runs_every_example(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.llm.resolve_api_key", lambda: None)
    at = AppTest.from_file(str(ROOT / "app_pages" / "4_Ask_the_Book.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    assert any("not investment advice" in w.value for w in at.warning)
    assert any(i.value.startswith("Demo mode") for i in at.info)
    for index in range(5):
        at.button(key=f"page_example_{index}").click().run()
        assert not at.exception, at.exception
    banners = [m.value for m in at.markdown if "ask-banner" in m.value]
    assert len(banners) == 5 and all("AI-generated" in b for b in banners)
    assert sum("Refused" in b for b in banners) == 2
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 5
    assert list(at.dataframe[0].value["Status"]) == ["refused", "refused", "answered", "answered", "answered"]

    at.selectbox(key="ask_focus_client").set_value("Me Claire Fontaine, avocate").run()
    at.button(key="page_example_0").click().run()
    assert any("“this client” is **Me Claire Fontaine" in m.value for m in at.markdown)


def test_meeting_brief_page_guards_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.llm.resolve_api_key", lambda: None)
    at = AppTest.from_file(str(ROOT / "app_pages" / "2_Meeting_Brief.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    assert any("not investment advice" in w.value for w in at.warning)
    assert any("How this AI works" in s.value for s in at.subheader)

    at.text_area(key="advisor_note").set_value("ping me at ada@example.com").run()
    at.button(key="generate_brief").click().run()
    assert not at.exception, at.exception
    assert any("email" in e.value.lower() for e in at.error)
    assert not any(w.value.startswith("AI-generated draft") for w in at.warning)

    at.text_area(key="advisor_note").set_value("").run()
    at.button(key="generate_brief").click().run()
    assert not at.exception, at.exception
    assert any(w.value.startswith("AI-generated draft") for w in at.warning)
    assert len(at.download_button) == 2
    assert all(button.disabled for button in at.download_button)

    at.button(key="mark_reviewed").click().run()
    assert not at.exception, at.exception
    assert len(at.download_button) == 2
    assert all(not button.disabled for button in at.download_button)


def test_risk_monitor_renders_for_every_client() -> None:
    at = AppTest.from_file(str(ROOT / "app_pages" / "1_Risk_Monitor.py"), default_timeout=60).run()
    assert not at.exception
    assert any("not investment advice" in w.value for w in at.warning)
    assert any("Yousif Bag" in m.value for m in at.markdown)
    assert any("Contribution to risk" in c.value for c in at.caption)
    select = at.selectbox[0]
    for name in select.options:
        select.set_value(name.split("  ·  ")[0]).run()
        assert not at.exception, name


def test_market_map_page_shows_the_fictional_example(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("geocoder should use the local cache")

    monkeypatch.setattr("core.market_map.fetch_commune", fail)
    at = AppTest.from_file(str(ROOT / "app_pages" / "3_Market_Map.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("not investment advice" in w.value for w in at.warning)
    assert any("Fictional example" in i.value for i in at.info)
    assert any("Methodology & limits" in s.value for s in at.subheader)
    assert any("Yousif Bag" in m.value and "LinkedIn" in m.value for m in at.markdown)
