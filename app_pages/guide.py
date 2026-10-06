"""Welcome window, guided tour and page help shared by every page (UI only, not a page)."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

import streamlit as st

from app_pages.ask_ui import ASK_PAGE, ASK_TITLE, CLIENT_KEYS

TOUR_CLIENT = "Me Claire Fontaine, avocate"
TOUR_QUESTION = "Which clients would face a margin call if equities fell 15%?"


@dataclass(frozen=True)
class TourStep:
    page: str
    path: str
    summary: str
    look_at: str


TOUR_STEPS: tuple[TourStep, ...] = (
    TourStep(
        "Risk Monitor",
        "app_pages/1_Risk_Monitor.py",
        "**Risk Monitor**: see her Lombard margin-call warning",
        "Me Fontaine's client view is open. Read the **Lombard** alert, then the **Equities -20%** stress test: "
        "her loan would hit a margin call.",
    ),
    TourStep(
        "Meeting Brief",
        "app_pages/2_Meeting_Brief.py",
        "**Meeting Brief**: generate her pre-meeting briefing",
        "Me Fontaine is selected. Press **Generate brief**: her alerts and figures come straight from the "
        "Risk Monitor, and export unlocks only after your review.",
    ),
    TourStep(
        ASK_TITLE,
        ASK_PAGE,
        f"**Ask**: “{TOUR_QUESTION}”",
        f"Click the example **{TOUR_QUESTION}** The answer cites the stress-test tool and the figures it returned.",
    ),
)

STYLE = """
<style>
.st-key-tour_banner {
  background: #f5f7fa; border: 1px solid #e2e8f0; border-left: 3px solid #1e3a5f;
  border-radius: 6px; padding: 0.7rem 1rem;
}
.st-key-tour_banner [data-testid="stVerticalBlock"] { gap: 0.2rem; }
.st-key-tour_banner [data-testid="stElementContainer"]:has([data-testid="stHtml"]) { display: none; }
.st-key-tour_banner [data-testid="stMarkdown"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
.st-key-tour_banner [data-testid="stMarkdown"] p { font-size: 0.9rem; color: #334155; margin: 0; }
.tour-label { color: #1e3a5f; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.st-key-tour_link button { padding: 0; min-height: 0; }
.st-key-tour_link button p { color: #1e3a5f; font-weight: 500; text-decoration: underline; text-underline-offset: 3px; }
.page-purpose { color: #475569; font-size: 1.02rem; margin: -0.35rem 0 0 0; }
@media (max-width: 640px) {
  .st-key-tour_banner { flex-direction: column; align-items: stretch; gap: 0.6rem; }
  .st-key-tour_banner > * { flex-basis: auto !important; }
}
</style>
"""
# A page with a root chat input sticks to the bottom, up to seconds after it renders; the banner
# sits at the top. Hold the top for a few seconds, until the visitor scrolls or types.
SCROLL_TO_TOP = """
<script>
(() => {
  let ticks = 0;
  const events = ["wheel", "touchstart", "keydown", "mousedown"];
  const pin = () => {
    const main = document.querySelector("section.stMain");
    if (main && main.scrollTop !== 0) main.scrollTop = 0;
  };
  const stop = () => {
    clearInterval(timer);
    document.removeEventListener("scroll", pin, true);
    events.forEach((name) => window.removeEventListener(name, stop, true));
  };
  const timer = setInterval(() => {
    pin();
    if (++ticks >= 80) stop();
  }, 125);
  document.addEventListener("scroll", pin, true);
  events.forEach((name) => window.addEventListener(name, stop, true));
})();
</script>
"""


def tour_step() -> TourStep | None:
    index = st.session_state.get("tour_step")
    return TOUR_STEPS[index] if isinstance(index, int) and 0 <= index < len(TOUR_STEPS) else None


def tour_page() -> str | None:
    """The page of the current step, or the page the tour was just ended on, so its layout stays put."""
    step = tour_step()
    return step.page if step else st.session_state.get("tour_ended_on")


def _go_to(index: int) -> None:
    step = TOUR_STEPS[index]
    st.session_state.tour_step = index
    st.session_state.tour_scroll_top = True
    st.session_state.pop("tour_ended_on", None)
    st.session_state[CLIENT_KEYS[step.page]] = TOUR_CLIENT
    st.switch_page(step.path)


def end_tour(stay_on: str | None = None) -> None:
    st.session_state.pop("tour_step", None)
    st.session_state.pop("tour_scroll_top", None)
    if stay_on:
        st.session_state.tour_ended_on = stay_on


def _close_welcome() -> None:
    st.session_state.welcome_open = False


@st.dialog("Welcome — a 2-minute tour", width="medium", on_dismiss=_close_welcome)
def welcome() -> None:
    st.markdown(
        "Follow one fictional client, **Me Claire Fontaine**, a lawyer with a Lombard loan, across three pages:"
    )
    st.markdown("\n".join(f"{number}. {step.summary}" for number, step in enumerate(TOUR_STEPS, start=1)))
    st.caption("Her name stays selected from one page to the next. You can leave the tour at any time.")
    with st.container(horizontal=True, gap="small"):
        if st.button("Start the tour", type="primary", key="tour_start"):
            _close_welcome()
            _go_to(0)
        if st.button("Explore freely", key="tour_skip"):
            _close_welcome()
            st.rerun()


def render_welcome() -> None:
    """The tour link. The window opens by itself on the first visit to Home, then only from the link."""
    if not st.session_state.get("welcome_seen"):
        st.session_state.welcome_seen = True
        st.session_state.welcome_open = True
    if st.button("Take the 2-minute tour", type="tertiary", key="tour_link"):
        st.session_state.welcome_open = True
    if st.session_state.get("welcome_open"):
        welcome()


def render_tour_banner(page: str) -> None:
    """The step banner, on the step's own page only. Opening any other page ends the tour."""
    if st.session_state.get("tour_ended_on") != page:
        st.session_state.pop("tour_ended_on", None)
    step = tour_step()
    if step is None:
        return
    if step.page != page:
        end_tour()
        return
    index = st.session_state.tour_step
    last = index == len(TOUR_STEPS) - 1
    with st.container(key="tour_banner", horizontal=True, vertical_alignment="center", gap="medium"):
        with st.container(width="stretch", gap=None):
            st.markdown(
                f'<div class="tour-label">Tour · Step {index + 1} of {len(TOUR_STEPS)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(step.look_at)
            # Inside the banner, so the banner keeps its place in the page and replaces the previous one.
            if st.session_state.pop("tour_scroll_top", False):
                st.html(SCROLL_TO_TOP, unsafe_allow_javascript=True)
        with st.container(horizontal=True, width="content", gap="small"):
            if st.button("Finish the tour" if last else "Next step", type="primary", key="tour_next"):
                if last:
                    end_tour(stay_on=page)
                    st.rerun()
                _go_to(index + 1)
            if not last and st.button("End tour", type="tertiary", key="tour_end"):
                end_tour(stay_on=page)
                st.rerun()


def page_intro(purpose: str, steps: Sequence[str]) -> None:
    """One line under the title on what the page is for, and a closed how-to."""
    st.markdown(f'<p class="page-purpose">{html.escape(purpose)}</p>', unsafe_allow_html=True)
    with st.expander("How to use this page"):
        st.markdown("\n".join(f"{number}. {item}" for number, item in enumerate(steps, start=1)))
