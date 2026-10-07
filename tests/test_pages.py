"""Smoke tests: every page renders for every client without raising (offline)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app_pages.guide import TOUR_CLIENT
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
    assert any(s.value == "What this project demonstrates" for s in at.subheader)
    assert not any(s.value in {"The problem", "The solution"} for s in at.subheader)
    assert not any("Placeholder" in c.value for c in at.caption)


def test_welcome_window_opens_once_and_from_the_link() -> None:
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not at.exception, at.exception
    assert at.button(key="tour_start").label == "Start the tour"
    assert at.button(key="tour_skip").label == "Explore freely"
    at.button(key="tour_skip").click().run()
    assert not any(b.key == "tour_start" for b in at.button)
    at.run()
    assert not any(b.key == "tour_start" for b in at.button)
    at.button(key="tour_link").click().run()
    assert any(b.key == "tour_start" for b in at.button)


def test_tour_keeps_me_fontaine_across_three_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.llm.resolve_api_key", lambda: None)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()

    def banner() -> str:
        return " ".join(m.value for m in at.markdown if "tour-label" in m.value)

    def next_from(path: str) -> None:
        # AppTest forgets the page after a switch_page; a browser keeps it in the URL.
        at.switch_page(path)
        at.button(key="tour_next").click().run()
        assert not at.exception, at.exception

    at.button(key="tour_start").click().run()
    assert not at.exception, at.exception
    assert "Step 1 of 3" in banner()
    assert at.selectbox(key="rm_client").value == TOUR_CLIENT

    next_from("app_pages/1_Risk_Monitor.py")
    assert "Step 2 of 3" in banner()
    assert at.selectbox(key="client").value == TOUR_CLIENT

    next_from("app_pages/2_Meeting_Brief.py")
    assert "Step 3 of 3" in banner()
    assert at.selectbox(key="ask_focus_client").value == TOUR_CLIENT
    assert at.button(key="tour_next").label == "Finish the tour"

    next_from("app_pages/4_Ask_the_Book.py")
    assert banner() == ""
    assert "tour_step" not in at.session_state


def test_leaving_the_tour_page_ends_the_tour() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.button(key="tour_start").click().run()
    assert any("Step 1 of 3" in m.value for m in at.markdown)
    at.switch_page("app_pages/3_Market_Map.py").run()
    assert not at.exception, at.exception
    assert "tour_step" not in at.session_state
    assert not any("tour-label" in m.value for m in at.markdown)


@pytest.mark.parametrize(
    "page", ["1_Risk_Monitor.py", "2_Meeting_Brief.py", "3_Market_Map.py", "4_Ask_the_Book.py"]
)
def test_every_page_explains_itself(page: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.llm.resolve_api_key", lambda: None)
    monkeypatch.setattr("core.market_map.fetch_commune", lambda *_a, **_k: None)
    at = AppTest.from_file(str(ROOT / "app_pages" / page), default_timeout=90).run()
    assert not at.exception, at.exception
    assert any("page-purpose" in m.value for m in at.markdown)
    help_panel = [e for e in at.expander if e.label == "How to use this page"]
    assert len(help_panel) == 1
    steps = [line for line in help_panel[0].markdown[0].value.splitlines() if line[:1].isdigit()]
    assert 3 <= len(steps) <= 4


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


def test_demo_examples_are_open_and_follow_a_free_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.llm.resolve_api_key", lambda: None)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not any(e.label == "Example questions" for e in at.expander)
    assert any(b.key == "fab_example_0" for b in at.button)
    at.chat_input(key="ask_fab_input").set_value("Which clients hold Apple?").run()
    assert not at.exception, at.exception
    assert any(m.value.startswith("Demo mode answers only the example questions") for m in at.markdown)
    at.button(key="fab_after_0_example_0").click().run()
    assert not at.exception, at.exception
    assert any(e.label.startswith("Tools used (1)") for e in at.expander)


def test_api_failure_switches_the_session_to_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def broken_model(_key: str):
        def call(*_: object) -> None:
            calls.append(1)
            raise RuntimeError("credit balance too low")

        return call

    monkeypatch.setattr("core.llm.resolve_api_key", lambda: "sk-test")
    monkeypatch.setattr("core.agent.anthropic_model", broken_model)
    at = AppTest.from_file(str(ROOT / "app_pages" / "4_Ask_the_Book.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    assert any(b.key == "page_example_0" for b in at.button)
    at.chat_input(key="ask_page_input").set_value("Which clients hold Apple?").run()
    assert not at.exception, at.exception
    assert not at.error
    assert any(m.value.startswith("Sorry, the live assistant is unavailable") for m in at.markdown)
    assert any(b.key == "page_after_0_example_0" for b in at.button)
    assert any("unavailable right now" in i.value for i in at.info)
    at.button(key="page_example_0").click().run()
    assert not at.exception, at.exception
    assert len(calls) == 1
    assert any("Demo mode" in m.value for m in at.markdown if "ask-banner" in m.value)


def test_live_mode_still_shows_the_examples_in_the_ask_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.llm.resolve_api_key", lambda: "sk-test")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    assert not any(e.label == "Example questions" for e in at.expander)
    assert any(b.key == "fab_example_0" for b in at.button)


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
