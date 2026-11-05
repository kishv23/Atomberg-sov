"""Competitive-intelligence dashboard for the SoV agent.

    streamlit run src/app.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

OUT = os.environ.get("SOV_OUT_DIR", "out")
st.set_page_config(page_title="Share-of-Voice Intelligence", layout="wide")


@st.cache_data(ttl=60)
def load(out_dir: str, stamp: float):
    def j(name):
        p = os.path.join(out_dir, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    def c(name):
        p = os.path.join(out_dir, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    return j("summary.json"), j("insights.json") or [], c("scored.csv"), c("aspects.csv")


def outputs_stamp(out_dir: str) -> float:
    p = os.path.join(out_dir, "summary.json")
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("Controls")
    if st.button("Re-run analysis on collected data"):
        from sov.analyze import analyze, load_config
        from sov.insights import generate_insights
        from sov.io import write_json
        import glob

        cfg = load_config("config.yaml")
        with st.spinner("Scoring…"):
            summary = analyze(glob.glob(os.path.join(cfg["output"]["data_dir"], "*.csv")), cfg, OUT)
            if summary:
                write_json(generate_insights(summary), os.path.join(OUT, "insights.json"))
        st.cache_data.clear()
    live = st.toggle("Live refresh (60s)", value=False, help="Reloads when the agent writes new results")

summary, insights, scored, aspects = load(OUT, outputs_stamp(OUT))
if summary is None or scored.empty:
    st.error("No results yet. Run `sov run` (crawl) or `sov analyze` (sample data) first.")
    st.stop()

primary = summary["primary"]
brands = summary["brands"]
platforms = sorted(scored["platform"].dropna().unique())
with st.sidebar:
    chosen = st.multiselect("Platforms", platforms, default=platforms)
    st.caption(f"{summary['total_items']} items · {summary['items_with_brand']} mention a tracked brand")
    st.caption("Queries: " + ", ".join(summary.get("queries", [])))

scored = scored[scored["platform"].isin(chosen)]

st.title(f"{primary} vs {len(brands) - 1} competitors: Share of Voice")

# ----------------- KPIs -----------------
cols = st.columns(4)
labels = {"rms": "Mention share", "wsov": "Weighted SoV", "sopv": "Share of positive voice"}
for col, row in zip(cols, summary["benchmark"]):
    col.metric(
        labels[row["metric"]],
        f"{row['primary_share']:.1%}",
        "leader" if row["primary_rank"] == 1 else f"#{row['primary_rank']}, {row['gap_to_leader']:.1%} behind {row['leader']}",
        delta_color="normal" if row["primary_rank"] == 1 else "inverse",
    )
cols[3].metric("Items analyzed", summary["total_items"])

# ----------------- Insights -----------------
st.subheader("What the agent found")
icons = {"risk": "🔴", "gap": "🟠", "watch": "🟡", "strength": "🟢"}
for i in insights[:8]:
    st.markdown(f"{icons.get(i['kind'], '•')} {i['text']}")

# ----------------- Benchmark -----------------
st.subheader("Competitive benchmark")
bench = pd.DataFrame(
    {
        "brand": brands,
        "Mention share": [summary["rms"]["share"][b] for b in brands],
        "Weighted SoV": [summary["wsov"]["share"][b] for b in brands],
        "Positive voice": [summary["sopv"]["share"][b] for b in brands],
    }
)
long = bench.melt("brand", var_name="metric", value_name="share")
st.altair_chart(
    alt.Chart(long)
    .mark_bar()
    .encode(
        y=alt.Y("brand:N", sort="-x", title=None),
        x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("metric:N", title=None),
        yOffset="metric:N",
        tooltip=["brand", "metric", alt.Tooltip("share:Q", format=".1%")],
    )
    .properties(height=420),
    use_container_width=True,
)

# ----------------- Aspects -----------------
st.subheader("Aspect-level sentiment")
if aspects.empty:
    st.info("No aspect mentions found yet.")
else:
    a = aspects[aspects["mentions"] >= 2]
    st.altair_chart(
        alt.Chart(a)
        .mark_rect()
        .encode(
            x=alt.X("aspect:N", title=None),
            y=alt.Y("brand:N", title=None),
            color=alt.Color("net_sentiment:Q", scale=alt.Scale(domain=[-1, 1], scheme="redyellowgreen"), title="Net sentiment"),
            tooltip=["brand", "aspect", "mentions", "positive", "negative", alt.Tooltip("net_sentiment:Q", format=".2f")],
        )
        .properties(height=320),
        use_container_width=True,
    )

# ----------------- Trend & platforms -----------------
left, right = st.columns(2)
with left:
    st.subheader("Mentions over time")
    t = scored.assign(published_at=pd.to_datetime(scored["published_at"], errors="coerce", utc=True)).dropna(subset=["published_at"])
    t = t[t["dominant_brand"].fillna("") != ""]
    if t.empty:
        st.caption("No dated items for the selected platforms.")
    else:
        t = t.groupby([pd.Grouper(key="published_at", freq="MS"), "dominant_brand"]).size().reset_index(name="items")
        st.altair_chart(
            alt.Chart(t).mark_line(point=True).encode(x="published_at:T", y="items:Q", color="dominant_brand:N"),
            use_container_width=True,
        )
with right:
    st.subheader("Brand presence by platform")
    h = scored[scored["dominant_brand"].fillna("") != ""].groupby(["platform", "dominant_brand"]).size().reset_index(name="items")
    st.altair_chart(
        alt.Chart(h).mark_rect().encode(x="platform:N", y="dominant_brand:N", color="items:Q", tooltip=["platform", "dominant_brand", "items"]),
        use_container_width=True,
    )

# ----------------- Evidence -----------------
st.subheader("Top publishers")
st.dataframe(pd.DataFrame(summary["top_publishers"]), use_container_width=True, hide_index=True)

with st.expander("Raw scored items"):
    st.dataframe(
        scored[["platform", "query", "rank", "title", "publisher", "dominant_brand", "sentiment_label", "engagement", "url"]],
        use_container_width=True,
        hide_index=True,
    )

if live:
    time.sleep(60)
    st.rerun()
