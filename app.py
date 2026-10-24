from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Marketing Performance Dashboard",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "marketing_raw.csv"

MEASURES = [
    "impressions",
    "clicks",
    "conversions",
    "engagements",
    "spend",
    "revenue",
]


def divide(numerator, denominator):
    """Return NaN when a ratio cannot be calculated."""
    if denominator == 0:
        return np.nan
    return numerator / denominator


def add_kpis(table):
    """Calculate weighted KPIs from aggregated counts."""
    table = table.copy()

    impressions = table["impressions"].replace(0, np.nan)
    clicks = table["clicks"].replace(0, np.nan)
    conversions = table["conversions"].replace(0, np.nan)
    spend = table["spend"].replace(0, np.nan)

    table["CTR (%)"] = table["clicks"] / impressions * 100
    table["Conversion rate (%)"] = table["conversions"] / clicks * 100
    table["CPC (INR)"] = table["spend"] / clicks
    table["CPA (INR)"] = table["spend"] / conversions
    table["ROAS"] = table["revenue"] / spend
    table["ROI proxy (%)"] = (
        (table["revenue"] - table["spend"]) / spend * 100
    )

    return table


def format_value(value, prefix="", suffix=""):
    if pd.isna(value):
        return "N/A"
    return f"{prefix}{value:,.2f}{suffix}"


def csv_bytes(table):
    return table.to_csv(index=False).encode("utf-8")


st.title("Marketing Performance Dashboard")
st.caption(
    "Academic demonstration | Synthetic data | Currency: INR"
)

if not RAW_PATH.exists():
    st.error("Dataset missing. Run: python generate_data.py")
    st.stop()

# ---------- Load and clean ----------
raw = pd.read_csv(RAW_PATH)
df = raw.copy()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

for column in ["channel", "campaign"]:
    df[column] = df[column].astype("string").str.strip()
    df[column] = df[column].replace("", pd.NA)

for column in MEASURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")

before_duplicates = len(df)
df = df.drop_duplicates(subset=["date", "channel", "campaign"])
duplicates_removed = before_duplicates - len(df)

valid = df[["date", "channel", "campaign"] + MEASURES].notna().all(axis=1)
valid &= df[MEASURES].ge(0).all(axis=1)
valid &= df["clicks"].le(df["impressions"])
valid &= df["conversions"].le(df["clicks"])

invalid_removed = int((~valid).sum())
df = df.loc[valid].copy().sort_values("date")

if df.empty:
    st.error("No valid records remain after cleaning.")
    st.stop()

df.to_csv(DATA_DIR / "marketing_clean.csv", index=False)

# Save cleaned data to SQLite and read it back for analysis.
with sqlite3.connect(DATA_DIR / "marketing.db") as connection:
    df.to_sql("campaign_performance", connection, if_exists="replace", index=False)
    df = pd.read_sql_query(
        "SELECT * FROM campaign_performance",
        connection,
        parse_dates=["date"],
    )

# ---------- Filters ----------
st.sidebar.header("Filter data")

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(df["date"].min().date(), df["date"].max().date()),
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date(),
)

channels = sorted(df["channel"].unique())
campaigns = sorted(df["campaign"].unique())

selected_channels = st.sidebar.multiselect(
    "Channels", channels, default=channels
)
selected_campaigns = st.sidebar.multiselect(
    "Campaigns", campaigns, default=campaigns
)

if len(selected_dates) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start_date, end_date = selected_dates

filtered = df.loc[
    df["date"].dt.date.between(start_date, end_date)
    & df["channel"].isin(selected_channels)
    & df["campaign"].isin(selected_campaigns)
].copy()

if filtered.empty:
    st.warning("No records match these filters. Adjust your selections.")
    st.stop()

totals = filtered[MEASURES].sum()

ctr = divide(totals["clicks"], totals["impressions"]) * 100
conversion_rate = divide(totals["conversions"], totals["clicks"]) * 100
roas = divide(totals["revenue"], totals["spend"])
roi = divide(
    totals["revenue"] - totals["spend"], totals["spend"]
) * 100

cards = st.columns(4)
cards[0].metric("Spend", format_value(totals["spend"], "₹"))
cards[1].metric("Revenue", format_value(totals["revenue"], "₹"))
cards[2].metric("Conversions", f'{int(totals["conversions"]):,}')
cards[3].metric("ROAS", format_value(roas, suffix="×"))

cards = st.columns(4)
cards[0].metric("Impressions", f'{int(totals["impressions"]):,}')
cards[1].metric("CTR", format_value(ctr, suffix="%"))
cards[2].metric("Conversion rate", format_value(conversion_rate, suffix="%"))
cards[3].metric("Marketing ROI proxy", format_value(roi, suffix="%"))

channel_summary = add_kpis(
    filtered.groupby("channel", as_index=False)[MEASURES].sum()
)

campaign_summary = add_kpis(
    filtered.groupby("campaign", as_index=False)[MEASURES].sum()
)

daily = filtered.groupby("date", as_index=False)[MEASURES].sum()

monthly_data = filtered.assign(
    month=filtered["date"].dt.strftime("%Y-%m")
)
monthly_summary = add_kpis(
    monthly_data.groupby("month", as_index=False)[MEASURES].sum()
)

overview_tab, campaigns_tab, reports_tab, quality_tab = st.tabs([
    "Overview",
    "Campaign analysis",
    "Reports",
    "Data quality",
])

with overview_tab:
    st.subheader("Daily spend and revenue")
    trend = daily.melt(
        id_vars="date",
        value_vars=["spend", "revenue"],
        var_name="metric",
        value_name="Amount (INR)",
    )
    st.plotly_chart(
        px.line(trend, x="date", y="Amount (INR)", color="metric"),
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Channel revenue")
        st.plotly_chart(
            px.bar(channel_summary, x="channel", y="revenue", color="channel"),
            use_container_width=True,
        )

    with right:
        st.subheader("Channel ROAS")
        st.plotly_chart(
            px.bar(channel_summary, x="channel", y="ROAS", color="channel"),
            use_container_width=True,
        )

    st.subheader("Engagement trend")
    st.plotly_chart(
        px.line(daily, x="date", y="engagements"),
        use_container_width=True,
    )

with campaigns_tab:
    st.subheader("Campaign performance")
    st.dataframe(campaign_summary.round(2), hide_index=True)

    st.plotly_chart(
        px.bar(
            campaign_summary,
            x="campaign",
            y="conversions",
            color="campaign",
        ),
        use_container_width=True,
    )

    st.subheader("Channel comparison")
    st.dataframe(channel_summary.round(2), hide_index=True)

with reports_tab:
    st.subheader("Insights for the selected filters")

    ranked_channels = channel_summary.dropna(subset=["ROAS"])
    ranked_campaigns = campaign_summary.dropna(subset=["ROAS"])

    insights = [
        f"Selected period: {start_date} to {end_date}.",
        f"Selected channels: {', '.join(selected_channels)}.",
        f"Selected campaigns: {', '.join(selected_campaigns)}.",
        f"Total spend: INR {totals['spend']:,.2f}.",
        f"Total revenue: INR {totals['revenue']:,.2f}.",
    ]

    if not ranked_channels.empty:
        best = ranked_channels.loc[ranked_channels["ROAS"].idxmax()]
        weakest = ranked_channels.loc[ranked_channels["ROAS"].idxmin()]
        insights.extend([
            f"Highest-ROAS channel: {best['channel']} "
            f"({best['ROAS']:.2f}x).",
            f"Lowest-ROAS channel: {weakest['channel']} "
            f"({weakest['ROAS']:.2f}x).",
            "Recommendation: investigate the lowest-ROAS channel's "
            "targeting, creative, and landing page before changing budgets.",
        ])

    if not ranked_campaigns.empty:
        best_campaign = ranked_campaigns.loc[
            ranked_campaigns["ROAS"].idxmax()
        ]
        insights.append(
            f"Highest-ROAS campaign: {best_campaign['campaign']} "
            f"({best_campaign['ROAS']:.2f}x)."
        )

    insights.extend([
        "Validate promising channels with controlled budget experiments.",
        "These are descriptive findings from synthetic data.",
        "ROI proxy excludes product costs and other operating expenses.",
    ])

    for insight in insights:
        st.write(f"- {insight}")

    st.subheader("Monthly report")
    st.dataframe(monthly_summary.round(2), hide_index=True)

    st.download_button(
        "Download campaign report",
        csv_bytes(campaign_summary),
        "campaign_report.csv",
        "text/csv",
    )
    st.download_button(
        "Download channel report",
        csv_bytes(channel_summary),
        "channel_report.csv",
        "text/csv",
    )
    st.download_button(
        "Download monthly report",
        csv_bytes(monthly_summary),
        "monthly_report.csv",
        "text/csv",
    )
    st.download_button(
        "Download business summary",
        "\n".join(insights),
        "business_summary.txt",
        "text/plain",
    )

with quality_tab:
    st.subheader("Cleaning audit — full dataset")
    st.write({
        "Raw rows": len(raw),
        "Duplicate rows removed": duplicates_removed,
        "Invalid or incomplete rows removed": invalid_removed,
        "Clean rows": len(df),
    })

    st.write(
        "Cleaning trims category spaces, parses dates and numbers, "
        "removes duplicate daily campaign records, and excludes missing "
        "or inconsistent measurements. Missing revenue is not assumed "
        "to mean zero sales."
    )

    st.subheader("Filtered clean records")
    st.dataframe(filtered, hide_index=True)

    st.download_button(
        "Download filtered clean data",
        csv_bytes(filtered),
        "filtered_marketing_data.csv",
        "text/csv",
    )

st.caption(
    "ROAS = revenue / advertising spend. "
    "ROI proxy = (revenue − advertising spend) / advertising spend × 100. "
    "Undefined ratios are displayed as N/A."
)