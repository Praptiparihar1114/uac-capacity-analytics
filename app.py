from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.kpi_engine import compute_all_kpis, load_features

# ----------------------------------------------------------------------
# Page config - must be the first Streamlit call
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="UAC System Capacity & Care Load Analytics",
    page_icon="\U0001F3E5",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Theme injection
# ----------------------------------------------------------------------
def load_css(path: str = "assets/style.css") -> None:
    css_path = Path(path)
    if css_path.exists():
        st.markdown("<style>" + css_path.read_text() + "</style>", unsafe_allow_html=True)


load_css()


# ----------------------------------------------------------------------
# Data loading (cached so the dashboard stays snappy)
# ----------------------------------------------------------------------
@st.cache_data
def get_data() -> pd.DataFrame:
    df = load_features(Path("data/processed/uac_features.csv"))
    return df


try:
    df = get_data()
except FileNotFoundError:
    st.error(
        "Processed data not found. Run 'python src/data_cleaning.py' and "
        "'python src/feature_engineering.py' first, in that order."
    )
    st.stop()

reported_df = df.dropna(subset=["total_system_load"])
DATA_MIN_DATE = reported_df["date"].min().date()
DATA_MAX_DATE = reported_df["date"].max().date()


@st.cache_data
def get_forecast() -> tuple[pd.DataFrame, dict]:
    forecast_df = pd.read_csv(Path("data/processed/uac_forecast.csv"), parse_dates=["date"])
    metrics = json.loads(Path("data/processed/forecast_metrics.json").read_text())
    return forecast_df, metrics


@st.cache_data
def get_quality_report() -> dict:
    return json.loads(Path("data/processed/data_quality_report.json").read_text())


# ----------------------------------------------------------------------
# Header - the "case file" signature element
# ----------------------------------------------------------------------
header_html = (
    '<div class="case-file-header">'
    '<div class="case-file-eyebrow">U.S. Dept. of Health &amp; Human Services &mdash; UAC Program</div>'
    '<p class="case-file-title">System Capacity &amp; Care Load Analytics</p>'
    '<p class="case-file-subtitle">'
    'Monitoring the CBP &rarr; HHS unaccompanied children care pipeline: '
    'intake, custody load, transfers, and discharge capacity.'
    '</p>'
    '</div>'
)
st.markdown(header_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar - filters
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")

    st.markdown('<div class="section-label">Date Range</div>', unsafe_allow_html=True)
    date_range = st.date_input(
        "Select reporting period",
        value=(DATA_MIN_DATE, DATA_MAX_DATE),
        min_value=DATA_MIN_DATE,
        max_value=DATA_MAX_DATE,
        label_visibility="collapsed",
    )
    # date_input can briefly return a single date while the user is picking
    # the second end of the range - guard against that instead of crashing.
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = DATA_MIN_DATE, DATA_MAX_DATE

    st.markdown('<div class="section-label">Time Granularity</div>', unsafe_allow_html=True)
    granularity = st.radio(
        "Granularity",
        options=["Daily", "Weekly", "Monthly"],
        index=0,
        label_visibility="collapsed",
        horizontal=True,
    )

    st.markdown('<div class="section-label">Metrics to Display</div>', unsafe_allow_html=True)
    show_cbp = st.checkbox("CBP Custody Load", value=True)
    show_hhs = st.checkbox("HHS Care Load", value=True)
    show_regime_bands = st.checkbox("Stress/Relief shading", value=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    footer_html = (
        '<div style="font-family: var(--font-mono); font-size: 0.72rem; '
        'color: #C7D2DE;">Data source: HHS/CBP daily UAC program export '
        '&middot; ' + str(DATA_MIN_DATE) + ' to ' + str(DATA_MAX_DATE) + '</div>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)

# Scope the working dataframe to the selected range - everything below
# (KPIs, charts, tables) reads from scoped_df / scoped_reported.
scoped_df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]
scoped_reported = scoped_df.dropna(subset=["total_system_load"])


# ----------------------------------------------------------------------
# KPI Summary Cards
# ----------------------------------------------------------------------
st.markdown('<div class="section-label">KPI Summary</div>', unsafe_allow_html=True)

kpis = compute_all_kpis(df, start_date, end_date)

# Current system regime (for card border coloring) = regime on the last
# reported day within the selected range.
current_regime = (
    scoped_reported["system_regime"].iloc[-1] if not scoped_reported.empty else "Normal"
)
regime_class_map = {"Stress": "regime-stress", "Relief": "regime-relief", "Normal": "regime-normal"}

kpi_cols = st.columns(len(kpis))
for col, kpi in zip(kpi_cols, kpis):
    trend_class = kpi.trend if kpi.trend in ("up", "down", "flat") else "flat"
    delta_html = ""
    if kpi.delta_pct is not None:
        arrow = {"up": "&#9650;", "down": "&#9660;", "flat": "&#8212;"}[trend_class]
        delta_html = (
            '<span class="kpi-delta ' + trend_class + '">' + arrow + ' '
            + "{:.1f}".format(abs(kpi.delta_pct)) + '%</span>'
        )

    value_display = "{:,.2f}".format(kpi.value) if abs(kpi.value) < 10 else "{:,.0f}".format(kpi.value)

    card_html = (
        '<div class="kpi-card ' + regime_class_map.get(current_regime, "regime-normal") + '">'
        '<div class="kpi-label">' + kpi.name + '</div>'
        '<div class="kpi-value">' + value_display + '</div>'
        '<div class="kpi-unit">' + kpi.unit + '</div>'
        + delta_html +
        '</div>'
    )

    with col:
        st.markdown(card_html, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ------------------------------------------------------------------------
# Shared chart helpers
# ------------------------------------------------------------------------
import plotly.graph_objects as go

REGIME_COLORS = {"Stress": "#B0413E", "Relief": "#3F7859", "Normal": "#6B7785"}
CHART_FONT = dict(family="IBM Plex Sans, sans-serif", color="#16202A", size=13)
MONO_FONT = dict(family="IBM Plex Mono, monospace", color="#16202A", size=12)


def resample_for_granularity(frame: pd.DataFrame, granularity: str) -> pd.DataFrame:
    # Daily stays as-is; Weekly/Monthly average the numeric columns over
    # the period so trends read cleanly at each zoom level.
    if granularity == "Daily":
        return frame
    freq = "W" if granularity == "Weekly" else "ME"
    numeric_cols = frame.select_dtypes(include="number").columns
    resampled = (
        frame.set_index("date")[numeric_cols].resample(freq).mean().reset_index()
    )
    return resampled


def get_regime_segments(frame: pd.DataFrame) -> list[dict]:
    # Collapse the day-by-day regime column into contiguous
    # (start, end, regime) segments, used to shade Stress/Relief windows on
    # the trend charts regardless of the granularity selected.
    if frame.empty:
        return []
    f = frame[["date", "system_regime"]].copy()
    f["group"] = (f["system_regime"] != f["system_regime"].shift()).cumsum()
    segments = []
    for _, seg in f.groupby("group"):
        regime = seg["system_regime"].iloc[0]
        if regime == "Normal":
            continue
        segments.append(
            {"start": seg["date"].min(), "end": seg["date"].max(), "regime": regime}
        )
    return segments


def apply_regime_shading(fig: go.Figure, frame: pd.DataFrame) -> go.Figure:
    for seg in get_regime_segments(frame):
        fig.add_vrect(
            x0=seg["start"],
            x1=seg["end"],
            fillcolor=REGIME_COLORS[seg["regime"]],
            opacity=0.10,
            line_width=0,
        )
    return fig


def base_layout(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(family="IBM Plex Serif, serif", size=16, color="#0F2A47")),
        font=CHART_FONT,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", gridcolor="#E3E7EB", showline=True, linecolor="#D7DBE0"),
        yaxis=dict(title=y_title, gridcolor="#E3E7EB", showline=True, linecolor="#D7DBE0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=20, l=10, r=10),
        height=420,
        hovermode="x unified",
    )
    return fig


# ------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------
tab_overview, tab_comparison, tab_trends, tab_forecast, tab_quality = st.tabs(
    [
        "System Load Overview",
        "CBP vs HHS Comparison",
        "Net Intake & Backlog",
        "Forecast",
        "Data Quality & Docs",
    ]
)

# ---------- Tab 1: System Load Overview ----------
with tab_overview:
    chart_df = resample_for_granularity(scoped_reported, granularity)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["date"], y=chart_df["total_system_load"],
            name="Total System Load", mode="lines",
            line=dict(color="#0F2A47", width=2),
        )
    )
    if "total_load_roll_avg_7d" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"], y=chart_df["total_load_roll_avg_7d"],
                name="7-day Rolling Avg", mode="lines",
                line=dict(color="#9C7A28", width=1.5, dash="dot"),
            )
        )
    if show_regime_bands:
        fig = apply_regime_shading(fig, scoped_reported)
    fig = base_layout(fig, "Total System Load Over Time", "Children under care")
    st.plotly_chart(fig, use_container_width=True)

    legend_html = (
        '<span class="regime-badge stress">Stress</span>&nbsp;'
        '<span class="regime-badge relief">Relief</span>&nbsp;'
        '<span class="regime-badge normal">Normal (unshaded)</span>'
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Growth Rate</div>', unsafe_allow_html=True)
    growth_fig = go.Figure()
    growth_fig.add_trace(
        go.Bar(
            x=chart_df["date"], y=chart_df["care_load_growth_rate_pct"],
            name="Care Load Growth Rate (%)",
            marker_color=chart_df["care_load_growth_rate_pct"].apply(
                lambda v: "#B0413E" if v > 0 else "#3F7859"
            ),
        )
    )
    growth_fig = base_layout(growth_fig, "Day-over-Day Care Load Growth Rate", "% change")
    growth_fig.update_layout(height=280)
    st.plotly_chart(growth_fig, use_container_width=True)

# ---------- Tab 2: CBP vs HHS Load Comparison ----------
with tab_comparison:
    chart_df = resample_for_granularity(scoped_reported, granularity)

    fig = go.Figure()
    if show_cbp:
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"], y=chart_df["cbp_active_load"],
                name="CBP Custody Load", mode="lines",
                line=dict(color="#3E6E8E", width=2),
                stackgroup="load" if show_hhs else None,
            )
        )
    if show_hhs:
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"], y=chart_df["hhs_active_load"],
                name="HHS Care Load", mode="lines",
                line=dict(color="#9C7A28", width=2),
                stackgroup="load" if show_cbp else None,
            )
        )
    fig = base_layout(fig, "CBP Custody vs HHS Care Load", "Children under care")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:1rem;">Daily Flow: Intake, Transfers &amp; Discharges</div>', unsafe_allow_html=True)
    flow_fig = go.Figure()
    flow_fig.add_trace(go.Bar(x=chart_df["date"], y=chart_df["cbp_intake"], name="CBP Intake", marker_color="#3E6E8E"))
    flow_fig.add_trace(go.Bar(x=chart_df["date"], y=chart_df["cbp_to_hhs_transfers"], name="Transfers to HHS", marker_color="#9C7A28"))
    flow_fig.add_trace(go.Bar(x=chart_df["date"], y=chart_df["hhs_discharges"], name="HHS Discharges", marker_color="#3F7859"))
    flow_fig = base_layout(flow_fig, "Daily Pipeline Flow", "Children")
    flow_fig.update_layout(barmode="group", height=350)
    st.plotly_chart(flow_fig, use_container_width=True)

# ---------- Tab 3: Net Intake & Backlog Trends ----------
with tab_trends:
    chart_df = resample_for_granularity(scoped_reported, granularity)

    st.markdown('<div class="section-label">Net Daily Intake</div>', unsafe_allow_html=True)
    intake_fig = go.Figure()
    intake_fig.add_trace(
        go.Bar(
            x=chart_df["date"], y=chart_df["net_daily_intake"],
            name="Net Daily Intake",
            marker_color=chart_df["net_daily_intake"].apply(
                lambda v: "#B0413E" if v > 0 else "#3F7859"
            ),
        )
    )
    intake_fig.add_trace(
        go.Scatter(
            x=chart_df["date"], y=chart_df["net_intake_roll_avg_7d"],
            name="7-day Rolling Avg", mode="lines",
            line=dict(color="#0F2A47", width=2, dash="dot"),
        )
    )
    intake_fig.add_hline(y=0, line_color="#4A5560", line_width=1)
    intake_fig = base_layout(intake_fig, "Net Daily Intake (Transfers In minus Discharges Out)", "Children/day")
    intake_fig.update_layout(height=350)
    st.plotly_chart(intake_fig, use_container_width=True)

    note_html = (
        '<div style="font-size:0.82rem; color:#4A5560; margin-top:-0.5rem;">'
        'Red bars = system filling faster than it empties that day. '
        'Green bars = system relieving load that day.'
        '</div>'
    )
    st.markdown(note_html, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Backlog Accumulation (14-day Rolling Sum)</div>', unsafe_allow_html=True)
    backlog_fig = go.Figure()
    backlog_fig.add_trace(
        go.Scatter(
            x=chart_df["date"], y=chart_df["backlog_rolling_sum_14d"],
            name="14-day Backlog Sum", mode="lines",
            line=dict(color="#0F2A47", width=2),
            fill="tozeroy",
            fillcolor="rgba(176, 65, 62, 0.15)",
        )
    )
    backlog_fig.add_hline(y=0, line_color="#4A5560", line_width=1)
    backlog_fig = base_layout(backlog_fig, "Backlog Indicator: Sustained Positive Net Intake", "Net children (14-day sum)")
    backlog_fig.update_layout(height=320)
    st.plotly_chart(backlog_fig, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Discharge Offset Ratio</div>', unsafe_allow_html=True)
    offset_fig = go.Figure()
    offset_fig.add_trace(
        go.Scatter(
            x=chart_df["date"], y=chart_df["discharge_offset_ratio"],
            name="Discharge Offset Ratio", mode="lines",
            line=dict(color="#3E6E8E", width=2),
        )
    )
    offset_fig.add_hline(
        y=1.0, line_color="#9C7A28", line_width=1.5, line_dash="dash",
        annotation_text="Parity (1.0)", annotation_position="top left",
    )
    offset_fig = base_layout(offset_fig, "Discharges vs Transfers Ratio Over Time", "Ratio")
    offset_fig.update_layout(height=320)
    st.plotly_chart(offset_fig, use_container_width=True)

    # Streak summary row
    if not scoped_reported.empty:
        longest_streak = int(scoped_reported["backlog_streak_days"].max())
        current_streak = int(scoped_reported["backlog_streak_days"].iloc[-1])
    else:
        longest_streak, current_streak = 0, 0

    streak_col1, streak_col2 = st.columns(2)
    with streak_col1:
        streak_html_1 = (
            '<div class="kpi-card regime-normal">'
            '<div class="kpi-label">Longest Backlog Streak in Range</div>'
            '<div class="kpi-value">' + str(longest_streak) + '</div>'
            '<div class="kpi-unit">consecutive days of positive net intake</div>'
            '</div>'
        )
        st.markdown(streak_html_1, unsafe_allow_html=True)
    with streak_col2:
        streak_html_2 = (
            '<div class="kpi-card regime-normal">'
            '<div class="kpi-label">Current Streak (last reported day)</div>'
            '<div class="kpi-value">' + str(current_streak) + '</div>'
            '<div class="kpi-unit">consecutive days of positive net intake</div>'
            '</div>'
        )
        st.markdown(streak_html_2, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Download Filtered Data</div>', unsafe_allow_html=True)
    csv_data = scoped_reported.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="uac_filtered_data.csv",
        mime="text/csv",
    )

# ---------- Tab 4: Forecast ----------
with tab_forecast:
    try:
        forecast_df, forecast_metrics = get_forecast()
    except FileNotFoundError:
        st.warning(
            "Forecast data not found. Run 'python src/forecasting.py' first, "
            "then reload this page."
        )
        forecast_df, forecast_metrics = None, None

    if forecast_df is not None:
        st.markdown('<div class="section-label">14-Day Forward Forecast: Total System Load</div>', unsafe_allow_html=True)

        # Show the trailing 60 real days for context, then the forecast.
        recent_history = reported_df.tail(60)

        forecast_fig = go.Figure()
        forecast_fig.add_trace(
            go.Scatter(
                x=recent_history["date"], y=recent_history["total_system_load"],
                name="Actual", mode="lines",
                line=dict(color="#0F2A47", width=2),
            )
        )
        forecast_fig.add_trace(
            go.Scatter(
                x=forecast_df["date"], y=forecast_df["predicted_total_system_load"],
                name="Forecast (Random Forest)", mode="lines+markers",
                line=dict(color="#9C7A28", width=2, dash="dash"),
                marker=dict(size=5),
            )
        )
        forecast_fig.add_vline(
            x=recent_history["date"].max(), line_color="#4A5560", line_width=1, line_dash="dot",
        )
        forecast_fig = base_layout(forecast_fig, "Total System Load: Recent History + 14-Day Forecast", "Children under care")
        forecast_fig.update_layout(height=420)
        st.plotly_chart(forecast_fig, use_container_width=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Model Evaluation (Holdout Backtest)</div>', unsafe_allow_html=True)

        model_metrics = forecast_metrics["model_metrics"]
        baseline_metrics = forecast_metrics["baseline_metrics"]

        eval_col1, eval_col2, eval_col3 = st.columns(3)
        metric_pairs = [
            ("MAE (lower is better)", "mae", ""),
            ("RMSE (lower is better)", "rmse", ""),
            ("MAPE (lower is better)", "mape_pct", "%"),
        ]
        for col, (label, key, suffix) in zip([eval_col1, eval_col2, eval_col3], metric_pairs):
            model_val = model_metrics[key]
            baseline_val = baseline_metrics[key]
            improvement_pct = ((baseline_val - model_val) / baseline_val) * 100 if baseline_val else 0

            metric_html = (
                '<div class="kpi-card regime-relief">'
                '<div class="kpi-label">' + label + '</div>'
                '<div class="kpi-value">' + str(model_val) + suffix + '</div>'
                '<div class="kpi-unit">baseline: ' + str(baseline_val) + suffix + '</div>'
                '<span class="kpi-delta down">&#9660; ' + "{:.0f}".format(improvement_pct) + '% better than baseline</span>'
                '</div>'
            )
            with col:
                st.markdown(metric_html, unsafe_allow_html=True)

        caption_html = (
            '<div style="font-size:0.8rem; color:#4A5560; margin-top:0.8rem;">'
            'Model: RandomForestRegressor with lag (1/2/3/7/14-day), rolling-average, '
            'and calendar features, evaluated on the last ' + str(forecast_metrics["holdout_days"])
            + ' real reporting days withheld from training. '
            'Baseline: ' + forecast_metrics["baseline_description"] + '.'
            '</div>'
        )
        st.markdown(caption_html, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Forecast Table &amp; Download</div>', unsafe_allow_html=True)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        forecast_csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download 14-day forecast as CSV",
            data=forecast_csv,
            file_name="uac_forecast_14day.csv",
            mime="text/csv",
        )

# ---------- Tab 5: Data Quality & Documentation ----------
with tab_quality:
    try:
        quality_report = get_quality_report()
    except FileNotFoundError:
        quality_report = None
        st.warning(
            "Data quality report not found. Run 'python src/data_cleaning.py' first, "
            "then reload this page."
        )

    if quality_report is not None:
        st.markdown('<div class="section-label">Data Preprocessing Summary</div>', unsafe_allow_html=True)

        qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
        summary_cards = [
            ("Total Calendar Days", str(quality_report["total_calendar_days"]), "Jan 2023 - Dec 2025 span"),
            ("Reported Days", str(quality_report["reported_days"]), "days HHS actually published data"),
            ("Missing Report Days", str(quality_report["missing_report_days"]), "gaps in the reporting schedule"),
            ("Anomalous Rows Flagged", str(quality_report["anomalous_rows_total"]), "logical constraint violations"),
        ]
        for col, (label, value, unit) in zip([qual_col1, qual_col2, qual_col3, qual_col4], summary_cards):
            card_html = (
                '<div class="kpi-card regime-normal">'
                '<div class="kpi-label">' + label + '</div>'
                '<div class="kpi-value">' + value + '</div>'
                '<div class="kpi-unit">' + unit + '</div>'
                '</div>'
            )
            with col:
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Methodology</div>', unsafe_allow_html=True)
        methodology_html = (
            '<div style="font-size:0.88rem; line-height:1.6; color:#16202A;">'
            '<b>1. Ingestion &amp; structuring</b> &mdash; raw HHS/CBP export parsed, '
            'thousands-separator strings (e.g. "2,484") converted to numeric, columns '
            'renamed to a consistent schema, sorted into strict chronological order.<br>'
            '<b>2. Calendar completion</b> &mdash; reindexed onto a complete daily calendar '
            'so reporting gaps are explicit missing values rather than silently skipped dates.<br>'
            '<b>3. Validation</b> &mdash; every row checked against two logical constraints: '
            'transfers out of CBP custody cannot exceed CBP\'s active load, and discharges from '
            'HHS care cannot exceed HHS\'s active load. Violations are flagged, not silently corrected, '
            'so the anomaly stays visible for analysis rather than being hidden by a "fix."<br>'
            '<b>4. Feature engineering</b> &mdash; Total System Load, Net Daily Intake, growth rate, '
            'rolling averages/volatility, backlog indicators, and a Stress/Relief/Normal regime '
            'classification (z-score of the 7-day rolling load against its trailing 90-day baseline) '
            'are derived on top of the cleaned data.'
            '</div>'
        )
        st.markdown(methodology_html, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Anomalous Rows in Selected Range</div>', unsafe_allow_html=True)
        anomalous_rows = scoped_df[scoped_df["is_anomalous"] == True]
        if anomalous_rows.empty:
            st.write("No logical constraint violations in the selected date range.")
        else:
            display_cols = [
                "date", "cbp_active_load", "cbp_to_hhs_transfers",
                "hhs_active_load", "hhs_discharges",
                "flag_transfers_exceed_cbp_load", "flag_discharges_exceed_hhs_load",
            ]
            st.dataframe(anomalous_rows[display_cols], use_container_width=True, hide_index=True)

        with st.expander("View sample of missing report dates"):
            missing_sample = quality_report.get("missing_dates_sample", [])
            if missing_sample:
                st.write(", ".join(missing_sample))
                st.caption(
                    "Showing first 25 of " + str(quality_report.get("missing_dates_total", 0))
                    + " total missing report dates."
                )
            else:
                st.write("No missing report dates found.")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Download Documentation</div>', unsafe_allow_html=True)
        quality_report_json = json.dumps(quality_report, indent=2)
        st.download_button(
            label="Download data quality report (JSON)",
            data=quality_report_json,
            file_name="uac_data_quality_report.json",
            mime="application/json",
        )