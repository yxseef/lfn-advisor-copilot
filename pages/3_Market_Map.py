"""LFN Market Map: firms in Geneva and Vaud, a prospect score, and a commune map (UI only)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core import APP_NAME, AUTHOR_CREDIT, DISCLAIMER
from core.market_map import (
    DEFAULT_WEIGHTS,
    FIRM_TYPES,
    TYPE_POINTS,
    FirmFileError,
    attach_coordinates,
    counts_by_commune,
    counts_by_type,
    creations_by_year,
    load_firm_book,
    score_firms,
    top_prospects,
)

TYPE_COLORS = {"lawyer": "#1e3a5f", "fiduciary": "#3b6ea5", "notary": "#64748b"}
TYPE_LABELS = {"lawyer": "Lawyer", "fiduciary": "Fiduciary", "notary": "Notary"}

st.set_page_config(page_title=f"Market Map · {APP_NAME}", layout="wide")
st.warning(DISCLAIMER, icon="⚠️")
st.title("LFN Market Map")
st.caption("Law, fiduciary and notary firms in Geneva and Vaud. A prospecting map, not a client list.")

try:
    book = load_firm_book()
except FirmFileError as exc:
    st.error(str(exc))
    st.stop()

if book.fictional:
    st.info(
        "Fictional example — these firms are invented so the demo runs without the commercial register. "
        "Every name starts with “Exemple”. Fill data/lfn_firms.csv to replace them."
    )
else:
    st.caption(
        "Source: data/lfn_firms.csv. Fields from a commercial-register extract only. "
        "Check the file before using the ranking."
    )

located = attach_coordinates(book.firms)

st.subheader("Filters")
filter_cols = st.columns(3)
canton_choice = filter_cols[0].multiselect("Canton", ["GE", "VD"], default=["GE", "VD"])
type_choice = filter_cols[1].multiselect(
    "Type",
    list(FIRM_TYPES),
    default=list(FIRM_TYPES),
    format_func=lambda kind: TYPE_LABELS[kind],
)
commune_options = sorted(located["commune"].unique())
commune_choice = filter_cols[2].multiselect("Commune", commune_options)

st.subheader("Prospect score")
st.caption(
    "Score = 100 × weighted average of four criteria, each scored from 0 to 1. "
    "It only orders who to contact first."
)
weight_cols = st.columns(4)
weight_help = {
    "legal_form": "SA and Sàrl score higher because those forms usually hold a firm account and can contract in the firm's name.",
    "seniority": "An older registration scores higher, capped at 30 years, as a rough sign that the firm is established.",
    "firm_type": "Fiduciaries rank first, then notaries, then lawyers, because their work sits closest to wealth planning, estates and firm treasury.",
    "local_density": "A commune with more LFN firms scores higher, so time goes where the profession is already concentrated.",
}
weight_labels = {
    "legal_form": "Legal form",
    "seniority": "Seniority",
    "firm_type": "Type",
    "local_density": "Local density",
}
weights: dict[str, float] = {}
for column, key in zip(weight_cols, DEFAULT_WEIGHTS):
    weights[key] = column.slider(weight_labels[key], 0.0, 3.0, float(DEFAULT_WEIGHTS[key]), 0.1)
    column.caption(weight_help[key])

with st.expander("Type points (also adjustable)"):
    st.caption("These are the points before the type weight. Set them equal if type should not change the order.")
    type_cols = st.columns(3)
    type_points = {
        kind: type_cols[index].slider(
            TYPE_LABELS[kind], 0.0, 1.0, float(TYPE_POINTS[kind]), 0.05, key=f"type_points_{kind}"
        )
        for index, kind in enumerate(FIRM_TYPES)
    }

scored = score_firms(located, weights=weights, type_points=type_points)
selected = scored[scored["canton"].isin(canton_choice) & scored["firm_type"].isin(type_choice)]
if commune_choice:
    selected = selected[selected["commune"].isin(commune_choice)]
selected = selected.reset_index(drop=True)

if selected.empty:
    st.info("No firm matches these filters.")
else:
    by_type = counts_by_type(selected)
    metrics = st.columns(4)
    metrics[0].metric("Firms", len(selected))
    for index, kind in enumerate(FIRM_TYPES, start=1):
        count = int(by_type.loc[by_type["firm_type"] == kind, "firms"].sum()) if not by_type.empty else 0
        metrics[index].metric(TYPE_LABELS[kind], count)

    st.subheader("Map")
    mappable = selected.dropna(subset=["lat_display", "lon_display"])
    if mappable.empty:
        st.warning("None of these communes are in the geocoding cache, so the map is empty. The table below still lists the firms.")
    else:
        shown = mappable.assign(type_label=mappable["firm_type"].map(TYPE_LABELS))
        fig = px.scatter_map(
            shown,
            lat="lat_display",
            lon="lon_display",
            color="type_label",
            hover_name="name",
            hover_data={
                "type_label": True,
                "legal_form": True,
                "commune": True,
                "canton": True,
                "status": True,
                "score": True,
                "lat_display": False,
                "lon_display": False,
            },
            color_discrete_map={TYPE_LABELS[kind]: TYPE_COLORS[kind] for kind in FIRM_TYPES},
            zoom=8,
            center={"lat": 46.35, "lon": 6.5},
            map_style="open-street-map",
            height=520,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Type")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Each point is the commune centre from swisstopo, not the firm's address. "
            "Firms in the same commune are offset by a few hundred metres so they stay visible."
        )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Firms by commune")
        by_commune = counts_by_commune(selected)
        fig = px.bar(
            by_commune,
            x="firms",
            y="commune",
            orientation="h",
            color_discrete_sequence=["#1e3a5f"],
        )
        fig.update_layout(
            height=420,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis={"categoryorder": "total ascending", "title": None},
            xaxis_title="Firms",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")
    with chart_right:
        st.subheader("Registrations by year")
        by_year = creations_by_year(selected)
        fig = px.bar(by_year, x="year", y="firms", color_discrete_sequence=["#3b6ea5"])
        fig.update_layout(
            height=420,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title=None,
            yaxis_title="Firms",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Year of the registration date in the file. On a Zefix extract that date is not a dedicated field.")

    st.subheader("Top 20 prospects")
    ranking = top_prospects(selected, 20)
    export = ranking.assign(
        type=ranking["firm_type"].map(TYPE_LABELS),
        rank=range(1, len(ranking) + 1),
        form_points=ranking["form_points"].round(2),
        seniority_points=ranking["seniority_points"].round(2),
        type_points=ranking["type_points"].round(2),
        density_points=ranking["density_points"].round(2),
    )[
        [
            "rank",
            "name",
            "type",
            "legal_form",
            "commune",
            "canton",
            "registered_on",
            "status",
            "score",
            "form_points",
            "seniority_points",
            "type_points",
            "density_points",
        ]
    ]
    st.dataframe(
        export.style.format(
            {
                "score": "{:.1f}",
                "form_points": "{:.2f}",
                "seniority_points": "{:.2f}",
                "type_points": "{:.2f}",
                "density_points": "{:.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download the top 20 (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="lfn_prospects_top20.csv",
        mime="text/csv",
    )

st.subheader("Methodology & limits")
st.markdown(
    """
**Source.** The public demo does not call Zefix. The federal API (Zefix PublicREST) answers
`401 Unauthorized` without HTTP Basic credentials, which are issued by the Federal Registry
of Commerce. Even with credentials, a search requires a name of at least three characters,
so it cannot list every firm in a canton in one call. The OpenAPI description is public and
is licensed as OGD: the source must be cited if those data are reused.

**What Zefix would return.** A search hit has the firm name, UID, legal form, commune
(legal seat), status (active, being cancelled, cancelled) and the date of the last SOGC
publication. The detailed record adds the purpose text and an address. There is no NOGA
code and no dedicated registration date. Profession is therefore classified from keywords
in the name and, when a file provides it, in the purpose: avocat, notaire, fiduciaire,
Treuhand, révision. The word "étude" alone is ignored, because it can be either a law office
or a notary office.

**This page.** While `data/lfn_firms.csv` is empty, the map uses an invented example.
Commune centres come from the swisstopo Search API on geo.admin.ch (layer gg25) and are
cached in `data/commune_geocodes.csv`. No website is scraped.

**Coverage.** Sole practitioners who are not in the commercial register are missing.
A keyword can miss a firm or label it as the wrong profession. Geneva and Vaud only.

**Score.** It is a weighted average of legal form, seniority (capped at 30 years),
profession and how many LFN firms sit in the same commune. The weights are the sliders
above. It is a calling order for an advisor, not a judgement of the firm, its clients
or its credit. Status is shown and is not part of the score. Density is computed on the
loaded book, then the filters are applied, so selecting one commune does not give every
firm a density of 1.
"""
)

st.divider()
st.caption(DISCLAIMER)
st.markdown(AUTHOR_CREDIT)
