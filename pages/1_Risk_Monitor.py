"""Risk Monitor: book-level risk dashboard and client drill-down (UI only)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import APP_NAME, AUTHOR_CREDIT, DISCLAIMER
from core.portfolios import ASSET_CLASS_LABELS, all_tickers, build_clients
from core.risk import (
    LOMBARD_WARNING_USAGE,
    PROFILE_LIMITS,
    WARNING_RATIO,
    ClientRisk,
    analyze_client,
    get_all_alerts,
    load_prices,
    overview_table,
)

st.set_page_config(page_title=f"Risk Monitor · {APP_NAME}", layout="wide")

STATUS_COLORS = {"breach": "#f8d7da", "warning": "#fff3cd", "ok": "#d1e7dd"}
# Blues and greys only. Red stays on alerts, the margin-call line, and stress losses.
DONUT_COLORS = (
    "#1e3a5f",
    "#3b6ea5",
    "#5b7c99",
    "#7d93a8",
    "#94a3b8",
    "#64748b",
    "#cbd5e1",
    "#334155",
)
SEVERITY_BOX = {"breach": st.error, "warning": st.warning, "info": st.info}


@st.cache_data(ttl=3600, show_spinner="Loading market data and computing risk…")
def load_book() -> tuple[list[ClientRisk], str, list[str], pd.Timestamp]:
    clients = build_clients()
    data = load_prices(all_tickers(clients))
    results = [analyze_client(c, data.prices) for c in clients]
    return results, data.source, data.missing, data.prices.index[-1]


def style_overview(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_color(row: pd.Series) -> list[str]:
        return [f"background-color: {STATUS_COLORS[row['Status']]}; color: #1f2937"] * len(row)

    # Streamlit renders NaN as "None" despite na_rep, so LTV is pre-formatted as text.
    shown = df.assign(LTV=df["LTV"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}"))
    return shown.style.apply(row_color, axis=1).format(
        {
            "AUM (CHF)": "{:,.0f}",
            "Volatility": "{:.1%}",
            "VaR 95% 1d (CHF)": "{:,.0f}",
            "VaR 1d (% AUM)": "{:.2%}",
            "Status": str.upper,
        }
    )


st.warning(DISCLAIMER, icon="⚠️")
st.title("Risk Monitor")
st.caption(
    "Fictional LFN client portfolios valued with real market data, in CHF. "
    "Risk figures use 3 years of daily history applied to today's holdings."
)

results, source, missing, as_of = load_book()
st.caption(f"Prices: **{source}** · last close **{as_of:%d %b %Y}**")
if missing:
    st.warning(f"No price data for: {', '.join(missing)}. Affected positions are excluded.")

tab_book, tab_client, tab_method = st.tabs(["Book overview", "Client view", "Methodology & limits"])

# ---------------------------------------------------------------------------
# Book overview
# ---------------------------------------------------------------------------
with tab_book:
    overview = overview_table(results)
    alerts = get_all_alerts(results)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients", len(results))
    c2.metric("Total AUM (CHF)", f"{overview['AUM (CHF)'].sum() / 1e6:,.1f} M")
    c3.metric("Limit breaches", int((alerts["severity"] == "breach").sum()))
    c4.metric("Warnings", int((alerts["severity"] == "warning").sum()))

    st.dataframe(style_overview(overview), hide_index=True, width="stretch")
    st.caption("Row colour = most severe alert: red = breach, amber = warning, green = no issue. "
               "LTV is shown only for clients with a Lombard loan.")

    st.subheader("All alerts")
    if alerts.empty:
        st.success("No alerts.")
    else:
        for _, a in alerts[alerts["severity"] != "info"].iterrows():
            SEVERITY_BOX[a["severity"]](f"**{a['client']}** · {a['category']} — {a['message']}")
        with st.expander("Informational alerts"):
            for _, a in alerts[alerts["severity"] == "info"].iterrows():
                st.info(f"**{a['client']}** · {a['category']} — {a['message']}")

# ---------------------------------------------------------------------------
# Client view
# ---------------------------------------------------------------------------
with tab_client:
    by_name = {r.client.name: r for r in results}
    name = st.selectbox(
        "Client",
        list(by_name),
        format_func=lambda n: f"{n}  ·  {by_name[n].status.upper()}",
    )
    r = by_name[name]
    c = r.client
    limits = PROFILE_LIMITS[c.risk_profile]

    st.markdown(
        f"**{c.client_type.capitalize()}** · {c.account_type} account · "
        f"**{c.risk_profile}** profile · reference currency {c.reference_currency}"
        + (f" · Lombard loan CHF {c.lombard.amount_chf:,.0f}" if c.lombard else "")
    )

    m = st.columns(4)
    m[0].metric("AUM (CHF)", f"{r.aum:,.0f}")
    m[1].metric("Volatility (ann.)", f"{r.volatility:.1%}", help=f"Profile limit {limits.max_volatility:.0%}")
    m[2].metric("Equities", f"{r.concentration.equity_weight:.0%}", help=f"Profile limit {limits.max_equity:.0%}")
    m[3].metric("LTV", f"{r.lombard.ltv:.0%}" if r.lombard else "—",
                help="Loan / market value of the pledged portfolio")
    m = st.columns(4)
    m[0].metric("VaR 95% 1d (CHF)", f"{r.var_1d:,.0f}", help="Historical simulation, 3 years")
    m[1].metric("VaR 95% 10d (CHF)", f"{r.var_10d:,.0f}", help="1-day VaR × √10")
    m[2].metric("ES 95% 1d (CHF)", f"{r.es_1d:,.0f}", help="Average loss on the worst 5 % of days")
    m[3].metric("ES 95% 10d (CHF)", f"{r.es_10d:,.0f}", help="1-day ES × √10")

    st.subheader("Alerts")
    if not r.alerts:
        st.success("No alerts: portfolio within all limits.")
    for a in r.alerts:
        SEVERITY_BOX[a.severity](f"**{a.category}** — {a.message}")

    left, right = st.columns(2)
    with left:
        st.subheader("Allocation")
        dim = st.radio("Group by", ["Asset class", "Currency"], horizontal=True, label_visibility="collapsed")
        if dim == "Asset class":
            alloc = r.concentration.by_asset_class.rename(index=ASSET_CLASS_LABELS)
        else:
            alloc = r.concentration.by_currency
        colors = [DONUT_COLORS[i % len(DONUT_COLORS)] for i in range(len(alloc))]
        fig = go.Figure(go.Pie(
            labels=alloc.index,
            values=alloc.values,
            hole=0.55,
            sort=False,
            textinfo="label+percent",
            marker=dict(colors=colors, line=dict(color="#ffffff", width=1)),
        ))
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Risk contributions")
        rc = r.contributions[r.contributions["weight"] > 0].head(12)
        rc_long = pd.DataFrame({
            "position": list(rc["name"]) * 2,
            "share": list(rc["weight"]) + list(rc["pct_of_risk"]),
            "measure": ["Weight"] * len(rc) + ["Contribution to risk"] * len(rc),
        })
        fig = px.bar(
            rc_long,
            x="share",
            y="position",
            color="measure",
            barmode="group",
            orientation="h",
            color_discrete_map={"Weight": "#94a3b8", "Contribution to risk": "#1e3a5f"},
        )
        fig.update_layout(
            height=420,
            margin=dict(t=10, b=90, l=10, r=10),
            xaxis_tickformat=".0%",
            yaxis={"categoryorder": "total ascending", "title": None},
            xaxis_title=None,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                x=0,
                title="Weight vs. Contribution to risk",
            ),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Weight is the position's share of portfolio value; Contribution to risk is its share of "
            "portfolio volatility, so a contribution above the weight means the position adds more "
            "volatility than its size suggests."
        )

    st.subheader("Value history (current holdings)")
    fig = go.Figure(go.Scatter(x=r.history.index, y=r.history.values, name="Portfolio value",
                               line=dict(color="#0f766e")))
    if r.lombard:
        # Market value at which usage reaches 100 %, assuming today's asset mix.
        fig.add_hline(y=r.aum * r.lombard.usage, line_dash="dash", line_color="#b91c1c",
                      annotation_text="Margin-call level", annotation_position="top left")
        fig.add_hline(y=r.lombard.loan, line_dash="dot", line_color="#64748b",
                      annotation_text="Loan", annotation_position="bottom left")
    fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="CHF", showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Hypothetical: today's quantities valued at past prices. Not the client's actual performance.")

    st.subheader("Stress tests")
    s = r.stress.reset_index()
    fig = px.bar(s, x="scenario", y="pnl_chf", text=s["pnl_pct"].map("{:.1%}".format),
                 color_discrete_sequence=["#b91c1c"])
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="P&L (CHF)", xaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    fmt = {"pnl_chf": "{:,.0f}", "pnl_pct": "{:.1%}", "value_after_chf": "{:,.0f}"}
    labels = {"pnl_chf": "P&L (CHF)", "pnl_pct": "P&L (%)", "value_after_chf": "Value after (CHF)"}
    if "lombard_usage_after" in r.stress.columns:
        fmt["lombard_usage_after"] = "{:.0%}"
        labels.update({"lombard_usage_after": "Lombard usage after", "margin_call": "Margin call"})
    st.dataframe(
        r.stress.rename(columns=labels).rename_axis("Scenario")
        .style.format({labels[k]: v for k, v in fmt.items()}),
        width="stretch",
    )

    if r.lombard:
        st.subheader("Lombard loan")
        lc = st.columns(5)
        lc[0].metric("Loan (CHF)", f"{r.lombard.loan:,.0f}")
        lc[1].metric("Lending value (CHF)", f"{r.lombard.lending_value:,.0f}")
        lc[2].metric("Usage", f"{r.lombard.usage:.0%}", help="Loan / lending value; ≥ 100 % = margin call")
        lc[3].metric("Free margin (CHF)", f"{r.lombard.available_margin:,.0f}")
        lc[4].metric("Cushion", f"{r.lombard.cushion:.0%}", help="Uniform market fall absorbable before a margin call")

    st.subheader("Profile compliance")
    comp = pd.DataFrame([{"Rule": k.rule, "Value": k.value, "Limit": k.limit, "Status": k.status.upper()}
                         for k in r.compliance])
    st.dataframe(
        comp.style.format({"Value": "{:.1%}", "Limit": "{:.0%}"}).apply(
            lambda row: [f"background-color: {STATUS_COLORS[row['Status'].lower()]}; color: #1f2937"] * len(row),
            axis=1,
        ),
        hide_index=True,
        width="stretch",
    )

    with st.expander("Positions"):
        pos = r.positions.assign(asset_class=r.positions["asset_class"].map(ASSET_CLASS_LABELS))
        st.dataframe(
            pos[["name", "asset_class", "currency", "quantity", "price_chf", "value_chf", "weight"]]
            .style.format({"quantity": "{:,.0f}", "price_chf": "{:,.2f}", "value_chf": "{:,.0f}", "weight": "{:.1%}"}),
            width="stretch",
        )

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
with tab_method:
    st.markdown(
        f"""
**Data.** Daily adjusted closes (Yahoo Finance, 3 years), converted to CHF with daily EUR/CHF and USD/CHF
rates. Cached for one hour; a local CSV is used if the download fails.

**Volatility.** Standard deviation of daily returns × √252.

**Historical VaR 95 %.** Loss not exceeded on 95 % of past days, applied to today's holdings.
10-day VaR = 1-day VaR × √10.

**Expected Shortfall 95 %.** Average loss on the worst 5 % of days.

**Risk contributions.** Euler decomposition: RCᵢ = wᵢ (Σw)ᵢ / σₚ; contributions sum to total volatility.

**Lombard.** Lending value = Σ market value × advance rate. Usage = loan / lending value.
Warning from {LOMBARD_WARNING_USAGE:.0%} usage, margin call above 100 %.

**Stress tests.** Instantaneous shocks: equities -20 %; EUR/CHF -10 % on EUR assets;
+200 bp parallel rate shift on bond ETFs (loss ≈ duration × 2 %).

**Alerts.** Breach above a limit; warning from {WARNING_RATIO:.0%} of a limit.
"""
    )
    lim = pd.DataFrame(
        {p: {"Equities max": l.max_equity, "Single equity max": l.max_single_security, "Gold max": l.max_gold,
             "Foreign currency max": l.max_non_chf, "Volatility max": l.max_volatility}
         for p, l in PROFILE_LIMITS.items()}
    )
    st.markdown("**Profile limits**")
    st.dataframe(lim.style.format("{:.0%}"), width="stretch")

st.divider()
st.caption(DISCLAIMER)
st.markdown(AUTHOR_CREDIT)
