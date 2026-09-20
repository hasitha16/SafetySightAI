from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.express as px


# ==================================================
# PROJECT SETUP
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# SAFESIGHT IMPORTS
# ==================================================

from agents.pattern_agent import detect_recurring_patterns
from agents.coordinator import (
    analyze_with_coordinator,
    ask_with_coordinator,
)
from feedback.feedback_manager import save_risk_override

# ==================================================
# FILE PATHS
# ==================================================

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "safesight_incidents_clean.csv"
)

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SafeSight AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM DESIGN
# ==================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background-color: #f6f8fb;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 4rem;
        max-width: 1500px;
    }


    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #101828;
        border-right: 1px solid #1d2939;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #344054;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.60rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.25rem;
    }


    /* ---------- Brand ---------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 11px;
        background: #1570ef;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 21px;
        font-weight: 700;
    }

    .brand-name {
        color: #101828 !important;
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -0.7px;
    }

    .brand-ai {
        color: #1570ef;
    }

    .system-status {
        display: inline-block;
        margin-top: 4px;
        padding: 5px 10px;
        background: #ecfdf3;
        color: #027a48;
        border: 1px solid #abefc6;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }


    /* ---------- Headings ---------- */

    .eyebrow {
        color: #1570ef;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .page-title {
        color: #101828 !important;
        font-size: 34px;
        line-height: 1.15;
        font-weight: 750;
        letter-spacing: -1px;
        margin: 0;
    }

    .page-description {
        color: #667085;
        font-size: 16px;
        line-height: 1.6;
        margin-top: 9px;
        margin-bottom: 28px;
        max-width: 760px;
    }

    .section-title {
        color: #101828;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .section-description {
        color: #667085;
        font-size: 14px;
        margin-bottom: 17px;
    }


    /* ---------- Metric Cards ---------- */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #eaecf0;
        padding: 20px 21px;
        border-radius: 12px;
        box-shadow:
            0 1px 2px rgba(16, 24, 40, 0.03);
        min-height: 120px;
    }

    div[data-testid="stMetricLabel"] {
        color: #667085;
        font-size: 13px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #101828;
        font-weight: 750;
    }


    /* ---------- Panels ---------- */

    .info-panel {
        background: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 12px;
        padding: 19px 21px;
        margin-top: 20px;
    }

    .info-panel-title {
        color: #344054;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .info-panel-text {
        color: #667085;
        font-size: 13px;
        line-height: 1.55;
        margin: 0;
    }




    /* ---------- Safety Action Center ---------- */
    .action-card { background:#fff; border:1px solid #eaecf0; border-radius:12px; padding:18px 19px; min-height:245px; box-shadow:0 1px 2px rgba(16,24,40,.03); }
    .action-priority { color:#175cd3; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.7px; margin-bottom:8px; }
    .action-title { color:#101828; font-size:16px; font-weight:750; line-height:1.35; margin-bottom:12px; }
    .action-label { color:#475467; font-size:11px; font-weight:750; text-transform:uppercase; letter-spacing:.45px; margin-top:10px; }
    .action-text { color:#344054; font-size:13px; line-height:1.5; margin-top:3px; }

    /* ---------- Signal Banner ---------- */

    .signal-card {
        background: #fffaeb;
        border: 1px solid #fedf89;
        border-radius: 12px;
        padding: 17px 20px;
        margin-top: 20px;
    }

    .signal-label {
        color: #b54708;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .6px;
    }

    .signal-title {
        color: #7a2e0e;
        font-size: 15px;
        font-weight: 700;
        margin-top: 4px;
    }

    .signal-text {
        color: #93370d;
        font-size: 13px;
        margin-top: 3px;
    }

    /* Force readable text in light dashboard area */

    .main .page-title,
    .main .section-title,
    .main .brand-name {
        color: #101828 !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #667085 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #101828 !important;
    }

    .main p {
        color: inherit;
}


    /* ---------- Readability ---------- */

    .page-description,
    .section-description,
    .info-panel-text {
        color: #475467 !important;
    }

    /* Streamlit expanders can inherit very light text in some themes. */
    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #d0d5dd;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary *,
    div[data-testid="stExpander"] details summary,
    div[data-testid="stExpander"] details summary * {
        color: #101828 !important;
        font-weight: 650 !important;
    }

    div[data-testid="stExpander"] svg {
        fill: #344054 !important;
        color: #344054 !important;
    }

    div[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] *,
    div[data-testid="stExpander"] p {
        color: #344054 !important;
    }

    /* Keep dataframe text readable in light mode. */
    div[data-testid="stDataFrame"] {
        color: #101828 !important;
    }


    /* ---------- Streamlit details ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    div[data-testid="stDecoration"] {
        display: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# DATA
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_FILE,
        dtype={"industry_code": str},
        low_memory=False
    )

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce"
    )

    return df


@st.cache_data
def load_patterns():

    df = load_data()

    return detect_recurring_patterns(df)


df = load_data()
patterns = load_patterns()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.markdown(
    """
    <div style="
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 2px;
    ">
        🛡️ SafeSight AI
    </div>

    <div style="
        font-size: 12px;
        color: #98a2b3;
        margin-bottom: 24px;
    ">
        Safety Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Emerging Precursors",
        "Analyze Report",
        "Safety Advisor"
    ],
    label_visibility="collapsed"
)


st.sidebar.divider()


st.sidebar.markdown(
    """
    <div style="
        font-size: 11px;
        color: #98a2b3;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        font-weight: 700;
    ">
        System
    </div>

    <div style="
        margin-top: 10px;
        font-size: 13px;
        color: #d0d5dd;
    ">
        <span style="color:#12b76a;">●</span>
        Analysis services online
    </div>

    <div style="
        margin-top: 7px;
        font-size: 12px;
        color: #98a2b3;
    ">
        105,991 historical reports
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# REUSABLE HEADER
# ==================================================

def page_header(
    eyebrow,
    title,
    description
):

    st.markdown(
        f"""
        <div class="eyebrow">
            {eyebrow}
        </div>

        <h1 class="page-title">
            {title}
        </h1>

        <div class="page-description">
            {description}
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    top_left, top_right = st.columns(
        [5, 1]
    )

    with top_left:
        st.markdown(
            '<div class="brand">'
            '<div class="brand-icon">S</div>'
            '<div class="brand-name">SafeSight <span class="brand-ai">AI</span></div>'
            '</div>',
            unsafe_allow_html=True
        )

    with top_right:

        st.markdown(
            """
            <div style="text-align:right;">
                <span class="system-status">
                    ● System online
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    page_header(
        "Safety overview",
        "See risks before they become incidents.",
        """
        SafeSight turns historical workplace safety reports
        into signals your team can understand and act on.
        """
    )


    # ----------------------------------------------
    # INTERACTIVE FILTERS
    # ----------------------------------------------

    st.markdown(
        """
        <div class="section-title">Explore the safety data</div>
        <div class="section-description">
            Filter the overview by state and year to investigate a specific operating context.
        </div>
        """,
        unsafe_allow_html=True
    )

    filter_left, filter_right = st.columns(2)

    state_options = ["All states"] + sorted(
        [str(value) for value in df["state"].dropna().unique()]
    )
    available_years = sorted(
        df["event_date"].dropna().dt.year.astype(int).unique().tolist(),
        reverse=True
    )
    year_options = ["All years"] + [str(year) for year in available_years]

    with filter_left:
        selected_state = st.selectbox(
            "State", state_options, key="overview_state_filter"
        )

    with filter_right:
        selected_year = st.selectbox(
            "Year", year_options, key="overview_year_filter"
        )

    filtered_df = df.copy()

    if selected_state != "All states":
        filtered_df = filtered_df[
            filtered_df["state"].astype(str) == selected_state
        ]

    if selected_year != "All years":
        filtered_df = filtered_df[
            filtered_df["event_date"].dt.year == int(selected_year)
        ]

    filtered_patterns = detect_recurring_patterns(filtered_df)

    st.caption(
        f"Viewing {len(filtered_df):,} reports · "
        f"{selected_state} · {selected_year}"
    )

    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------

    outcomes = filtered_patterns[
        "serious_outcomes"
    ]

    st.markdown(
        """
        <div class="section-title">
            Safety at a glance
        </div>

        <div class="section-description">
            Based on the selected historical incident view.
        </div>
        """,
        unsafe_allow_html=True
    )


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    with metric1:

        st.metric(
            "Reports analysed",
            f'{outcomes["total_reports"]:,}'
        )


    with metric2:

        st.metric(
            "Hospitalizations",
            f'{outcomes["hospitalized"]:,}'
        )


    with metric3:

        st.metric(
            "Amputations",
            f'{outcomes["amputations"]:,}'
        )


    with metric4:

        st.metric(
            "Hospitalization rate",
            f'{outcomes["hospitalization_rate"]:.1f}%'
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # ----------------------------------------------
    # CHARTS
    # ----------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            What the data is showing
        </div>

        <div class="section-description">
            Historical activity and the most frequently
            recorded incident types.
        </div>
        """,
        unsafe_allow_html=True
    )


    left, right = st.columns(
        [1, 1],
        gap="large"
    )


    # ----------------------------------------------
    # YEARLY TREND
    # ----------------------------------------------

    with left:

        st.markdown(
            "**Incident activity over time**"
        )

        yearly = (
            filtered_patterns["yearly_trends"]
            .copy()
        )

        yearly["year"] = (
            yearly["year"]
            .astype(str)
        )

        fig_year = px.line(
            yearly,
            x="year",
            y="report_count",
            markers=True,
            labels={
                "year": "",
                "report_count": "Reports"
            }
        )

        fig_year.update_traces(
            line=dict(
                width=3,
                color="#1570ef"
            ),
            marker=dict(
                size=7,
                color="#1570ef"
            )
        )

        fig_year.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            font=dict(
                color="#344054"
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(color="#344054", size=12),
                title_font=dict(color="#344054", size=13)
            ),
            yaxis=dict(
                gridcolor="#d0d5dd",
                tickfont=dict(color="#344054", size=12),
                title_font=dict(color="#344054", size=13)
            )
        )

        st.plotly_chart(
            fig_year,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    # ----------------------------------------------
    # TOP INCIDENTS
    # ----------------------------------------------

    with right:

        st.markdown(
            "**Most common incident types**"
        )

        incidents = (
            filtered_patterns["incident_types"]
            .head(7)
            .copy()
        )

        # Short labels make the dashboard easier
        # to read while hover still shows data.

        incidents["display_name"] = (
            incidents["incident_type"]
            .str.strip()
            .str.slice(0, 43)
        )

        incidents["display_name"] = (
            incidents["display_name"]
            .apply(
                lambda x:
                x + "..."
                if len(x) >= 43
                else x
            )
        )

        incidents = incidents.sort_values(
            "count"
        )

        fig_incidents = px.bar(
            incidents,
            x="count",
            y="display_name",
            orientation="h",
            custom_data=[
                "incident_type"
            ],
            labels={
                "count": "Reports",
                "display_name": ""
            }
        )

        fig_incidents.update_traces(
            marker_color="#2e90fa",
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Reports: %{x:,}"
                "<extra></extra>"
            )
        )

        fig_incidents.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            font=dict(
                color="#344054",
                size=11
            ),
            xaxis=dict(
                gridcolor="#d0d5dd",
                tickfont=dict(color="#344054", size=12),
                title_font=dict(color="#344054", size=13)
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color="#344054", size=12),
                title_font=dict(color="#344054", size=13)
            )
        )

        st.plotly_chart(
            fig_incidents,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    # ----------------------------------------------
    # EMERGING SIGNAL
    # ----------------------------------------------

    emerging = filtered_patterns.get(
        "emerging_incidents"
    )

    if (
        emerging is not None
        and len(emerging) > 0
    ):

        top_signal = (
            emerging.iloc[0]
        )

        incident_name = str(
            top_signal[
                "incident_type"
            ]
        ).strip()

        growth = top_signal[
            "growth_percent"
        ]

        recent_count = top_signal[
            "recent_count"
        ]

        previous_count = top_signal[
            "previous_count"
        ]


        st.markdown(
            (
                '<div class="signal-card">'
                '<div class="signal-label">'
                'Emerging signal'
                '</div>'
                '<div class="signal-title">'
                f'{incident_name}'
                '</div>'
                '<div class="signal-text">'
                'Reports increased from '
                f'<b>{previous_count}</b> to '
                f'<b>{recent_count}</b> '
                'in the latest comparison period '
                f'({growth:.1f}% increase). '
                'Review the Emerging Precursors workspace '
                'for context.'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True
        )



    # ----------------------------------------------
    # RISK HOTSPOT MATRIX
    # ----------------------------------------------

    st.markdown(
        """
        <div class="section-title" style="margin-top: 26px;">
            Risk Hotspot Matrix
        </div>
        <div class="section-description">
            Compare frequently recorded incident categories with serious historical outcomes
            in the selected view.
        </div>
        """,
        unsafe_allow_html=True
    )

    hotspot_source = filtered_df[
        filtered_df["event_type"].notna()
        & (filtered_df["event_type"].astype(str).str.strip() != "")
        & (filtered_df["event_type"].astype(str) != "Unknown")
    ].copy()

    if not hotspot_source.empty:
        hotspot = (
            hotspot_source.groupby("event_type", as_index=False)
            .agg(
                reports=("report_id", "count"),
                hospitalizations=("hospitalized", "sum"),
                amputations=("amputation", "sum"),
            )
        )

        hotspot["serious_outcomes"] = (
            hotspot["hospitalizations"] + hotspot["amputations"]
        )
        hotspot["serious_outcome_rate"] = (
            hotspot["serious_outcomes"] / hotspot["reports"] * 100
        ).round(1)

        # Focus on categories with meaningful report volume so tiny categories
        # do not dominate the visual.
        hotspot = hotspot.sort_values("reports", ascending=False).head(20)

        fig_hotspot = px.scatter(
            hotspot,
            x="reports",
            y="serious_outcome_rate",
            size="serious_outcomes",
            hover_name="event_type",
            custom_data=[
                "hospitalizations",
                "amputations",
                "serious_outcomes"
            ],
            labels={
                "reports": "Report frequency",
                "serious_outcome_rate": "Serious outcomes per 100 reports",
                "serious_outcomes": "Serious outcomes"
            }
        )

        fig_hotspot.update_traces(
            marker=dict(
                opacity=0.78,
                line=dict(width=1)
            ),
            hovertemplate=(
                "<b>%{hovertext}</b>"
                "<br>Reports: %{x:,}"
                "<br>Serious outcomes / 100 reports: %{y:.1f}"
                "<br>Hospitalizations: %{customdata[0]:,}"
                "<br>Amputations: %{customdata[1]:,}"
                "<extra></extra>"
            )
        )

        fig_hotspot.update_layout(
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=10, r=20, t=20, b=10),
            font=dict(color="#344054"),
            xaxis=dict(
                gridcolor="#d0d5dd",
                tickfont=dict(color="#344054", size=12),
                title_font=dict(color="#344054", size=13)
            ),
            yaxis=dict(
                gridcolor="#d0d5dd",
                tickfont=dict(color="#344054", size=12),
                title_font=dict(color="#344054", size=13)
            )
        )

        st.plotly_chart(
            fig_hotspot,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.caption(
            "Bubble size represents the number of recorded serious outcomes. "
            "This is a historical prioritization view, not a prediction of future risk."
        )
    else:
        st.info("No incident categories are available for the selected filters.")

    # ----------------------------------------------
    # EXPORT INVESTIGATION SUMMARY
    # ----------------------------------------------

    top_recurring_export = "Not available"
    incident_export = filtered_patterns.get("incident_types")
    if incident_export is not None and not incident_export.empty:
        recurring_row = incident_export.iloc[0]
        if "incident_type" in incident_export.columns:
            top_recurring_export = str(recurring_row["incident_type"])
        elif "event_type" in incident_export.columns:
            top_recurring_export = str(recurring_row["event_type"])
        else:
            top_recurring_export = str(recurring_row.iloc[0])

    emerging_export = filtered_patterns.get("emerging_incidents")
    top_emerging_export = "Not available"
    if emerging_export is not None and not emerging_export.empty:
        emerging_row = emerging_export.iloc[0]
        if "incident_type" in emerging_export.columns:
            top_emerging_export = str(emerging_row["incident_type"])
        elif "event_type" in emerging_export.columns:
            top_emerging_export = str(emerging_row["event_type"])
        else:
            top_emerging_export = str(emerging_row.iloc[0])

    export_summary = pd.DataFrame(
        [
            {
                "Selected state": selected_state,
                "Selected year": selected_year,
                "Reports analyzed": int(outcomes["total_reports"]),
                "Hospitalizations": int(outcomes["hospitalized"]),
                "Amputations": int(outcomes["amputations"]),
                "Hospitalization rate (%)": float(outcomes["hospitalization_rate"]),
                "Strongest emerging signal": top_emerging_export,
                "Most recurring incident": top_recurring_export,
                "Suggested review priority": (
                    "Investigate emerging and recurring hazards, verify existing "
                    "controls, and prioritize categories associated with serious outcomes."
                ),
            }
        ]
    )

    export_csv = export_summary.to_csv(index=False).encode("utf-8")

    st.markdown(
        """
        <div class="section-title" style="margin-top: 26px;">
            Investigation Export
        </div>
        <div class="section-description">
            Export the current filtered safety context for review, handover or follow-up.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.download_button(
        label="Download investigation summary",
        data=export_csv,
        file_name="safesight_investigation_summary.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # ----------------------------------------------
    # SAFETY ACTION CENTER
    # ----------------------------------------------
    st.markdown(
        """
        <div class="section-title" style="margin-top: 26px;">Safety Action Center</div>
        <div class="section-description">Evidence-led priorities to help a safety officer decide what to investigate next.</div>
        """,
        unsafe_allow_html=True
    )

    recurring = filtered_patterns["incident_types"].iloc[0]
    recurring_name = str(recurring["incident_type"]).strip()
    recurring_count = int(recurring["count"])
    action1, action2, action3 = st.columns(3, gap="medium")

    if emerging is not None and len(emerging) > 0:
        with action1:
            st.markdown(
                f"""<div class="action-card">
                <div class="action-priority">Priority 01 · Emerging</div>
                <div class="action-title">{incident_name}</div>
                <div class="action-label">Evidence</div>
                <div class="action-text">Reports changed from <b>{previous_count}</b> to <b>{recent_count}</b> ({growth:.1f}% increase) in the latest comparison period.</div>
                <div class="action-label">Suggested review</div>
                <div class="action-text">Investigate recent locations, tasks and controls associated with this signal.</div>
                </div>""",
                unsafe_allow_html=True
            )

    with action2:
        st.markdown(
            f"""<div class="action-card">
            <div class="action-priority">Priority 02 · Recurring</div>
            <div class="action-title">{recurring_name}</div>
            <div class="action-label">Evidence</div>
            <div class="action-text"><b>{recurring_count:,}</b> historical reports make this the most frequently recorded incident type in the dataset.</div>
            <div class="action-label">Suggested review</div>
            <div class="action-text">Review recurring work conditions and existing controls around this incident category.</div>
            </div>""",
            unsafe_allow_html=True
        )

    with action3:
        st.markdown(
            f"""<div class="action-card">
            <div class="action-priority">Priority 03 · Serious outcomes</div>
            <div class="action-title">High-consequence incident review</div>
            <div class="action-label">Evidence</div>
            <div class="action-text"><b>{outcomes['hospitalized']:,}</b> hospitalizations and <b>{outcomes['amputations']:,}</b> amputations are recorded in the historical dataset.</div>
            <div class="action-label">Suggested review</div>
            <div class="action-text">Prioritize investigation of recurring hazards associated with severe outcomes and verify critical controls.</div>
            </div>""",
            unsafe_allow_html=True
        )

    # ----------------------------------------------
    # RESPONSIBLE USE
    # ----------------------------------------------

    st.markdown(
        '<div class="info-panel">'
        '<div class="info-panel-title">About these signals</div>'
        '<p class="info-panel-text">'
        'SafeSight uses historical workplace incident data to surface recurring '
        'and changing safety patterns. These signals support human safety review '
        'and are not predictions that a future incident will occur.'
        '</p></div>',
        unsafe_allow_html=True
    )


# ==================================================
# EMERGING PRECURSORS
# ==================================================

elif page == "Emerging Precursors":

    page_header(
        "Early warning",
        "Signals worth attention.",
        """
        SafeSight compares recent incident activity with the
        preceding period to surface categories that are increasing.
        """
    )

    emerging = patterns["emerging_incidents"].copy()

    if emerging.empty:
        st.info("No emerging precursor signals were detected.")

    else:

        # ------------------------------------------
        # SUMMARY
        # ------------------------------------------

        top_signal = emerging.iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Signals detected",
                f"{len(emerging):,}"
            )

        with c2:
            st.metric(
                "Highest growth",
                f'{top_signal["growth_percent"]:.1f}%'
            )

        with c3:
            st.metric(
                "Recent reports",
                f'{int(top_signal["recent_count"]):,}'
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ------------------------------------------
        # TOP SIGNAL
        # ------------------------------------------

        signal_name = str(
            top_signal["incident_type"]
        ).strip()

        st.markdown(
            (
                '<div class="signal-card">'
                '<div class="signal-label">Strongest emerging signal</div>'
                f'<div class="signal-title">{signal_name}</div>'
                '<div class="signal-text">This category increased from '
                f'<b>{int(top_signal["previous_count"])}</b> reports in the previous '
                'comparison period to '
                f'<b>{int(top_signal["recent_count"])}</b> in the recent period — '
                f'an increase of <b>{top_signal["growth_percent"]:.1f}%</b>.'
                '</div></div>'
            ),
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ------------------------------------------
        # CHART
        # ------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                Fastest-growing incident categories
            </div>

            <div class="section-description">
                Percentage change between the recent and
                preceding comparison periods.
            </div>
            """,
            unsafe_allow_html=True
        )

        chart_data = emerging.head(10).copy()

        chart_data["display_name"] = (
            chart_data["incident_type"]
            .astype(str)
            .str.strip()
            .str.slice(0, 48)
        )

        chart_data["display_name"] = (
            chart_data["display_name"].apply(
                lambda x: x + "..."
                if len(x) >= 48
                else x
            )
        )

        chart_data = chart_data.sort_values(
            "growth_percent"
        )

        fig_growth = px.bar(
            chart_data,
            x="growth_percent",
            y="display_name",
            orientation="h",
            custom_data=[
                "incident_type",
                "previous_count",
                "recent_count"
            ],
            labels={
                "growth_percent": "Growth (%)",
                "display_name": ""
            }
        )

        fig_growth.update_traces(
            marker_color="#f79009",
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Previous: %{customdata[1]}"
                "<br>Recent: %{customdata[2]}"
                "<br>Growth: %{x:.1f}%"
                "<extra></extra>"
            )
        )

        fig_growth.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(
                l=10,
                r=20,
                t=20,
                b=10
            ),
            font=dict(
                color="#344054"
            ),
            xaxis=dict(
                gridcolor="#d0d5dd",
                title="Growth (%)",
                tickfont=dict(color="#344054", size=12),
                title_font=dict(color="#344054", size=13)
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color="#344054", size=12),
                title_font=dict(color="#344054", size=13)
            )
        )

        st.plotly_chart(
            fig_growth,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        # ------------------------------------------
        # COMPARISON TABLE
        # ------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                Signal details
            </div>

            <div class="section-description">
                Compare report volume before and after the
                detected increase.
            </div>
            """,
            unsafe_allow_html=True
        )

        table_data = emerging.head(10)[
            [
                "incident_type",
                "previous_count",
                "recent_count",
                "growth_percent"
            ]
        ].copy()

        table_data.columns = [
            "Incident category",
            "Previous period",
            "Recent period",
            "Growth %"
        ]

        table_data["Growth %"] = (
            table_data["Growth %"]
            .round(1)
        )

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # IMPORTANT CONTEXT
        # ------------------------------------------

        st.markdown(
            '<div class="info-panel">'
            '<div class="info-panel-title">How to interpret these signals</div>'
            '<p class="info-panel-text">'
            'A large percentage increase does not automatically mean a category is '
            'the most dangerous. Categories with small historical counts can show '
            'large percentage changes. SafeSight surfaces these changes for human '
            'investigation rather than treating them as predictions.'
            '</p></div>',
            unsafe_allow_html=True
        )

# ==================================================
# ANALYZE REPORT
# ==================================================

elif page == "Analyze Report":

    page_header(
        "Incident analysis",
        "Understand a report in seconds.",
        """
        Enter a free-text safety report. SafeSight extracts the
        hazard, identifies risk factors, assesses risk and retrieves
        related historical incidents.
        """
    )

    # --------------------------------------------------
    # REPORT INPUT
    # --------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            New safety observation
        </div>

        <div class="section-description">
            Describe what happened or paste an existing safety report.
        </div>
        """,
        unsafe_allow_html=True
    )

    report_text = st.text_area(
        "Safety report",
        height=180,
        placeholder=(
            "Example: A worker was cleaning material near a conveyor "
            "while the machine was still operating. Their glove came "
            "close to the moving roller..."
        ),
        label_visibility="collapsed"
    )

    analyze_clicked = st.button(
        "Analyze report →",
        type="primary"
    )

    # --------------------------------------------------
    # RUN MULTI-AGENT ANALYSIS
    # --------------------------------------------------

    if analyze_clicked:

        if not report_text.strip():

            st.warning(
                "Enter a safety report before running the analysis."
            )

        else:

            with st.spinner(
                "SafeSight agents are reviewing the report..."
            ):

                try:

                    result = analyze_with_coordinator(
                        report_text=report_text.strip(),
                        report_id="DASHBOARD"
                    )

                    st.session_state["latest_analysis"] = result
                    st.session_state["latest_report_text"] = (
                        report_text.strip()
                    )

                except Exception as error:

                    st.error(
                        "The analysis could not be completed."
                    )

                    st.caption(str(error))

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    if "latest_analysis" in st.session_state:

        result = st.session_state["latest_analysis"]

        if result.get("error"):

            error_message = str(result["error"])

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message                    or "quota" in error_message.lower()
            ):

                st.warning(
                    "AI analysis is temporarily unavailable because "
                    "the language-model request limit has been reached. "
                    "Please try again after the API quota resets."
                )

                st.caption(
                        "Historical analytics and the Safety Advisor "
                    "remain available."
                )

            else:

                st.error(
                    "The report analysis could not be completed."
                )

                with st.expander("Technical details"):
                    st.code(error_message)

        else:

            parsed = result.get(
                "parsed_report",
                {}
            )

            risk = result.get(
                "risk_assessment",
                {}
            )

            evidence = result.get(
                "rag_evidence",
                []
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()

            # ==========================================
            # RISK RESULT
            # ==========================================

            risk_level = str(
                risk.get(
                    "risk_level",
                    "UNKNOWN"
                )
            ).upper()

            try:
                confidence = float(
                    risk.get(
                        "confidence",
                        0
                    )
                )
            except (TypeError, ValueError):
                confidence = 0

            if risk_level == "HIGH":

                risk_background = "#fef3f2"
                risk_border = "#fecdca"
                risk_text = "#b42318"
                risk_message = (
                    "Immediate safety review recommended"
                )

            elif risk_level == "MEDIUM":

                risk_background = "#fffaeb"
                risk_border = "#fedf89"
                risk_text = "#b54708"
                risk_message = (
                    "Review and corrective action recommended"
                )

            elif risk_level == "LOW":

                risk_background = "#ecfdf3"
                risk_border = "#abefc6"
                risk_text = "#027a48"
                risk_message = (
                    "Monitor and maintain existing controls"
                )

            else:

                risk_background = "#f2f4f7"
                risk_border = "#d0d5dd"
                risk_text = "#344054"
                risk_message = (
                    "Risk assessment returned without a known label"
                )

            st.markdown(
                (
                    f'<div style="color:{risk_text};'
                    'font-size:12px;font-weight:800;'
                    'letter-spacing:1px;">'
                    f'{risk_level} RISK'
                    '</div>'
                    '<div style="color:#101828;'
                    'font-size:22px;font-weight:750;'
                    'margin-top:5px;">'
                    f'{risk_message}'
                    '</div>'
                    '<div style="color:#667085;'
                    'font-size:13px;margin-top:7px;">'
                    'Agent confidence: '
                    f'{confidence:.0%}'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

            # ==========================================
            # EXTRACTED INFORMATION
            # ==========================================

            st.markdown(
                """
                <div class="section-title">
                    What SafeSight found
                </div>

                <div class="section-description">
                    Structured safety information extracted
                    from the free-text report.
                </div>
                """,
                unsafe_allow_html=True
            )

            left, right = st.columns(
                2,
                gap="large"
            )

            # ------------------------------------------
            # LEFT COLUMN
            # ------------------------------------------

            with left:

                st.markdown(
                    "##### Hazard"
                )

                st.write(
                    parsed.get(
                        "hazard_type",
                        "Not identified"
                    )
                )

                st.markdown(
                    "##### Location context"
                )

                st.write(
                    parsed.get(
                        "location_context",
                        "Not identified"
                    )
                )

                st.markdown(
                    "##### Potential consequence"
                )

                st.write(
                    parsed.get(
                        "potential_consequence",
                        "Not identified"
                    )
                )

            # ------------------------------------------
            # RIGHT COLUMN
            # ------------------------------------------

            with right:

                st.markdown(
                    "##### Risk factors"
                )

                risk_factors = parsed.get(
                    "risk_factors",
                    []
                )

                if risk_factors:

                    for factor in risk_factors:
                        st.write(
                            f"• {factor}"
                        )

                else:

                    st.write(
                        "No specific risk factors identified."
                    )

                st.markdown(
                    "##### Equipment involved"
                )

                equipment = parsed.get(
                    "equipment_involved",
                    []
                )

                if equipment:

                    for item in equipment:
                        st.write(
                            f"• {item}"
                        )

                else:

                    st.write(
                        "No equipment identified."
                    )

                st.markdown(
                    "##### People at risk"
                )

                people = parsed.get(
                    "people_at_risk",
                    []
                )

                if people:

                    for person in people:
                        st.write(
                            f"• {person}"
                        )

                else:

                    st.write(
                        "No specific group identified."
                    )

            # ==========================================
            # RISK REASONING
            # ==========================================

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="section-title">
                    Risk assessment
                </div>

                <div class="section-description">
                    Why the agent assigned this level and
                    what action it recommends.
                </div>
                """,
                unsafe_allow_html=True
            )

            assessment_left, assessment_right = (
                st.columns(
                    2,
                    gap="large"
                )
            )

            with assessment_left:

                st.markdown(
                    "##### Why this risk level?"
                )

                st.write(
                    risk.get(
                        "reasoning",
                        "No reasoning available."
                    )
                )

            with assessment_right:

                st.markdown(
                    "##### Recommended action"
                )

                st.write(
                    risk.get(
                        "recommended_action",
                        "No recommendation available."
                    )
                )

            # ==========================================
            # HUMAN RISK OVERRIDE
            # ==========================================

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="section-title">
                    Human risk review
                </div>

                <div class="section-description">
                    A safety officer can correct the AI-assigned risk
                    level and record the reason for the decision.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                f"AI classification: {risk_level}"
            )

            corrected_risk = st.selectbox(
                "Corrected risk level",
                ["LOW", "MEDIUM", "HIGH"],
                index=(
                    ["LOW", "MEDIUM", "HIGH"].index(risk_level)
                    if risk_level in ["LOW", "MEDIUM", "HIGH"]
                    else 1
                ),
                key="corrected_risk"
            )

            override_reason = st.text_area(
                "Reason for correction",
                placeholder=(
                    "Example: The machine was operating without "
                    "isolation, creating a potential for serious injury."
                ),
                key="override_reason"
            )

            save_override_clicked = st.button(
                "Save risk override",
                key="save_risk_override"
            )

            if save_override_clicked:

                if corrected_risk == risk_level:

                    st.warning(
                        "Select a different risk level before "
                        "saving an override."
                    )

                elif not override_reason.strip():

                    st.warning(
                        "Enter a reason for the risk correction."
                    )

                else:

                    try:

                        save_risk_override(
                            report_id="DASHBOARD",
                            report_text=st.session_state.get(
                                "latest_report_text",
                                ""
                            ),
                            original_risk=risk_level,
                            corrected_risk=corrected_risk,
                            reason=override_reason.strip()
                        )

                        st.success(
                            f"Risk corrected from {risk_level} "
                            f"to {corrected_risk}. "
                            "The review has been logged."
                        )

                    except Exception as error:

                        st.error(
                            "The risk override could not be saved."
                        )

                        with st.expander("Technical details"):
                            st.code(str(error))


            # ==========================================
            # HISTORICAL RAG EVIDENCE
            # ==========================================

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class="section-title">
                    Related historical evidence
                </div>

                <div class="section-description">
                    Similar incidents retrieved from SafeSight's
                    historical vector store. These reports provide
                    evidence and context for the current review.
                </div>
                """,
                unsafe_allow_html=True
            )

            if evidence:

                for index, item in enumerate(
                    evidence[:3],
                    start=1
                ):

                    metadata = item.get("metadata", {}) or {}

                    # RAG evidence may expose metadata either as top-level
                    # fields or inside a nested metadata dictionary.
                    report_id = item.get("report_id", metadata.get("report_id", "Unknown"))
                    city = item.get("city", metadata.get("city", "Unknown"))
                    state = item.get("state", metadata.get("state", "Unknown"))
                    event_date = item.get("event_date", metadata.get("event_date", "Unknown"))
                    event_type = item.get("event_type", metadata.get("event_type", "Unknown"))
                    injury_nature = item.get("injury_nature", metadata.get("injury_nature", "Unknown"))
                    content = item.get("content", "")
                    with st.expander(
                        f"Incident {index}  •  "
                        f"{city.title()}, {state.title()}  •  "
                        f"Report {report_id}"
                    ):

                        st.markdown(
                            f"**Incident type:** {event_type}"
                        )

                        st.markdown(
                            f"**Date:** {event_date}"
                        )

                        st.markdown(
                            f"**Injury:** {injury_nature}"
                        )

                        st.write(content)

            else:

                st.info(
                    "No related historical evidence "
                    "was returned for this report."
                )

            # ==========================================
            # RESPONSIBLE USE
            # ==========================================

            st.markdown(
                '<div class="info-panel">'
                '<div class="info-panel-title">Human review remains in control</div>'
                '<p class="info-panel-text">'
                "SafeSight's AI-generated risk assessment supports safety officers by "
                'organizing information and retrieving historical evidence. Final safety '
                'decisions should follow workplace procedures and professional judgement.'
                '</p></div>',
                unsafe_allow_html=True
            )

# ==================================================
# SAFETY ADVISOR
# ==================================================

elif page == "Safety Advisor":

    page_header(
        "Safety intelligence",
        "Ask your safety data.",
        """
        Ask about recurring hazards, locations, emerging
        precursors or similar historical incidents.
        SafeSight automatically selects the right analysis tool.
        """
    )

    # --------------------------------------------------
    # EXAMPLE QUESTIONS
    # --------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            What would you like to investigate?
        </div>

        <div class="section-description">
            Ask a question in plain language. SafeSight will
            route it to the appropriate safety analysis tool.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Try: “What are the risks in Texas?” · "
        "“What incidents are increasing recently?” · "
        "“Show common patterns in the dataset.”"
    )

    officer_question = st.text_input(
        "Safety question",
        placeholder="e.g. What are the main safety risks in Texas?",
        label_visibility="collapsed"
    )

    ask_clicked = st.button(
        "Ask SafeSight →",
        type="primary"
    )

    # --------------------------------------------------
    # RUN LANGGRAPH TOOL AGENT
    # --------------------------------------------------

    if ask_clicked:

        if not officer_question.strip():

            st.warning(
                "Enter a safety question first."
            )

        else:

            with st.spinner(
                "SafeSight is selecting the right analysis tool..."
            ):

                try:

                    result = ask_with_coordinator(
                        officer_question.strip()
                    )

                    st.session_state[
                        "advisor_result"
                    ] = result

                    st.session_state[
                        "advisor_question"
                    ] = officer_question.strip()

                except Exception as error:

                    st.error(
                        "SafeSight could not process the question."
                    )

                    st.caption(str(error))

    # --------------------------------------------------
    # DISPLAY TOOL RESULT
    # --------------------------------------------------

    if "advisor_result" in st.session_state:

        result = st.session_state[
            "advisor_result"
        ]

        if result.get("error"):

            st.error(
                f'Analysis stopped: {result["error"]}'
            )

        else:

            response = result.get(
                "tool_response",
                {}
            )

            tool_name = response.get(
                "selected_tool",
                "Unknown tool"
            )

            tool_result = response.get(
                "result",
                {}
            )

            question_lower = (
                st.session_state
                .get("advisor_question", "")
                .lower()
            )

            # ==================================================
            # DETERMINE TOOL NAME + RESULT SHAPE
            # ==================================================

            # Format A:
            # {
            #     "selected_tool": "get_location_risk",
            #     "result": {...}
            # }
            if (
                isinstance(response, dict)
                and "selected_tool" in response
                and "result" in response
            ):

                tool_name = response["selected_tool"]
                tool_result = response["result"]


            # Format B:
            # {
            #     "location": "texas",
            #     "total_reports": 17490,
            #     ...
            # }
            elif (
                isinstance(response, dict)
                and "location" in response
                and "total_reports" in response
            ):

                tool_name = "get_location_risk"
                tool_result = response


            # Format C:
            # Emerging precursor list
            elif isinstance(response, list):

                tool_name = "get_emerging_precursors"
                tool_result = response


            # Format D:
            # Pattern analysis dictionary
            elif (
                isinstance(response, dict)
                and "serious_outcomes" in response
            ):

                tool_name = "analyze_safety_patterns"
                tool_result = response


            # Fallback:
            # Determine intended tool from question
            else:

                tool_result = response

                location_words = [
                    "texas",
                    "florida",
                    "new york",
                    "illinois",
                    "ohio",
                    "pennsylvania",
                    "georgia",
                    "wisconsin",
                    "alabama",
                    "missouri",
                    "houston",
                    "orlando",
                    "chicago",
                    "dallas",
                    "san antonio",
                    "miami",
                    "atlanta",
                    "denver",
                ]

                emerging_words = [
                    "increasing",
                    "emerging",
                    "growing",
                    "rising",
                    "precursor",
                    "recent trend",
                ]

                pattern_words = [
                    "overall",
                    "common",
                    "most common",
                    "dataset",
                    "historical trend",
                    "patterns",
                    "statistics",
                ]

                if any(
                    word in question_lower
                    for word in location_words
                ):
                    tool_name = "get_location_risk"

                elif any(
                    word in question_lower
                    for word in emerging_words
                ):
                    tool_name = "get_emerging_precursors"

                elif any(
                    word in question_lower
                    for word in pattern_words
                ):
                    tool_name = "analyze_safety_patterns"

                else:
                    tool_name = "search_similar_incidents"

            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------------------------------
            # ROUTING STATUS
            # ------------------------------------------

            st.markdown(
                (
                    '<div style="'
                    'background:#eff8ff;'
                    'border:1px solid #b2ddff;'
                    'border-radius:12px;'
                    'padding:16px 19px;'
                    'margin-bottom:22px;'
                    '">'
                    '<div style="'
                    'color:#175cd3;'
                    'font-size:11px;'
                    'font-weight:800;'
                    'text-transform:uppercase;'
                    'letter-spacing:0.8px;'
                    '">'
                    'LANGGRAPH TOOL ROUTING'
                    '</div>'
                    '<div style="'
                    'color:#101828;'
                    'font-size:16px;'
                    'font-weight:700;'
                    'margin-top:4px;'
                    '">'
                    f'{tool_name}'
                    '</div>'
                    '<div style="'
                    'color:#667085;'
                    'font-size:13px;'
                    'margin-top:3px;'
                    '">'
                    'SafeSight selected this tool automatically '
                    'based on the safety officer&apos;s question.'
                    '</div>'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

            # ==========================================
            # LOCATION RISK RESULT
            # ==========================================

            if tool_name == "get_location_risk":

                if isinstance(tool_result, dict):

                    location = str(
                        tool_result.get(
                            "location",
                            "Location"
                        )
                    ).title()

                    st.markdown(
                        f"""
                        <div class="section-title">
                            Safety profile · {location}
                        </div>

                        <div class="section-description">
                            Historical incident activity recorded
                            for this location.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Reports",
                            f'{tool_result.get("total_reports", 0):,}'
                        )

                    with c2:
                        st.metric(
                            "Hospitalizations",
                            f'{tool_result.get("hospitalized", 0):,}'
                        )

                    with c3:
                        st.metric(
                            "Amputations",
                            f'{tool_result.get("amputations", 0):,}'
                        )

                    top_types = tool_result.get(
                        "top_incident_types",
                        {}
                    )

                    if top_types:

                        st.markdown("<br>", unsafe_allow_html=True)

                        st.markdown(
                            """
                            <div class="section-title">
                                Most common incident types
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        location_df = pd.DataFrame(
                            list(top_types.items()),
                            columns=[
                                "Incident type",
                                "Reports"
                            ]
                        )

                        location_df = (
                            location_df
                            .sort_values("Reports")
                        )

                        fig_location = px.bar(
                            location_df,
                            x="Reports",
                            y="Incident type",
                            orientation="h"
                        )

                        fig_location.update_traces(
                            marker_color="#2e90fa"
                        )

                        fig_location.update_layout(
                            height=360,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                            margin=dict(
                                l=10,
                                r=10,
                                t=20,
                                b=10
                            ),
                            font=dict(
                                color="#344054"
                            ),
                            xaxis=dict(
                                gridcolor="#d0d5dd",
                                tickfont=dict(color="#344054", size=12),
                                title_font=dict(color="#344054", size=13)
                            ),
                            yaxis=dict(
                                showgrid=False,
                                tickfont=dict(color="#344054", size=12),
                                title_font=dict(color="#344054", size=13)
                            )
                        )

                        st.plotly_chart(
                            fig_location,
                            use_container_width=True,
                            config={
                                "displayModeBar": False
                            }
                        )

            # ==========================================
            # EMERGING PRECURSORS
            # ==========================================

            elif tool_name == "get_emerging_precursors":

                st.markdown(
                    """
                    <div class="section-title">
                        Emerging safety signals
                    </div>

                    <div class="section-description">
                        Incident categories showing increased
                        activity in the latest comparison period.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if isinstance(tool_result, list):

                    precursor_df = pd.DataFrame(
                        tool_result
                    )

                    if not precursor_df.empty:

                        columns = [
                            col for col in [
                                "incident_type",
                                "previous_count",
                                "recent_count",
                                "growth_percent"
                            ]
                            if col in precursor_df.columns
                        ]

                        display_df = precursor_df[
                            columns
                        ].copy()

                        rename_map = {
                            "incident_type":
                                "Incident category",
                            "previous_count":
                                "Previous period",
                            "recent_count":
                                "Recent period",
                            "growth_percent":
                                "Growth %"
                        }

                        display_df = (
                            display_df.rename(
                                columns=rename_map
                            )
                        )

                        if "Growth %" in display_df:

                            display_df[
                                "Growth %"
                            ] = (
                                display_df[
                                    "Growth %"
                                ].round(1)
                            )

                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True
                        )

            # ==========================================
            # OVERALL PATTERN ANALYSIS
            # ==========================================

            elif tool_name == "analyze_safety_patterns":

                st.markdown(
                    """
                    <div class="section-title">
                        Dataset-wide safety patterns
                    </div>

                    <div class="section-description">
                        Historical patterns across the complete
                        SafeSight incident dataset.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if isinstance(tool_result, dict):

                    outcomes = tool_result.get(
                        "serious_outcomes",
                        {}
                    )

                    if outcomes:

                        c1, c2, c3 = st.columns(3)

                        with c1:

                            st.metric(
                                "Reports",
                                f'{outcomes.get("total_reports", 0):,}'
                            )

                        with c2:

                            st.metric(
                                "Hospitalizations",
                                f'{outcomes.get("hospitalized", 0):,}'
                            )

                        with c3:

                            st.metric(
                                "Amputations",
                                f'{outcomes.get("amputations", 0):,}'
                            )

                    incident_types = tool_result.get(
                        "incident_types"
                    )

                    if isinstance(
                        incident_types,
                        pd.DataFrame
                    ):

                        st.markdown("<br>", unsafe_allow_html=True)

                        st.markdown(
                            "##### Most common incident types"
                        )

                        st.dataframe(
                            incident_types.head(10),
                            use_container_width=True,
                            hide_index=True
                        )

                    elif incident_types is not None:

                        st.markdown("<br>", unsafe_allow_html=True)

                        st.markdown(
                            "##### Most common incident types"
                        )

                        st.dataframe(
                            pd.DataFrame(
                                incident_types
                            ).head(10),
                            use_container_width=True,
                            hide_index=True
                        )

            # ==========================================
            # SIMILAR INCIDENT SEARCH
            # ==========================================

            elif tool_name == "search_similar_incidents":

                st.markdown(
                    """
                    <div class="section-title">
                        Similar historical incidents
                    </div>

                    <div class="section-description">
                        Relevant reports retrieved from the
                        SafeSight vector store.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if isinstance(tool_result, list):

                    for index, incident in enumerate(
                        tool_result,
                        start=1
                    ):

                        # search_similar_incidents returns metadata
                        # as top-level fields, not under "metadata".
                        content = incident.get(
                            "content",
                            ""
                        )

                        report_id = incident.get(
                            "report_id",
                            "Unknown"
                        )

                        city = incident.get(
                            "city",
                            "Unknown"
                        )

                        state = incident.get(
                            "state",
                            "Unknown"
                        )

                        event_type = incident.get(
                            "event_type",
                            "Unknown"
                        )

                        event_date = incident.get(
                            "event_date",
                            "Unknown"
                        )

                        injury_nature = incident.get(
                            "injury_nature",
                            "Unknown"
                        )

                        with st.expander(
                            f"Incident {index}  •  "
                            f"{city.title()}, {state.title()}  •  "
                            f"Report {report_id}"
                        ):

                            st.markdown(
                                f"**Incident type:** {event_type}"
                            )

                            st.caption(
                                f"Event date: {event_date}  •  "
                                f"Injury: {injury_nature}"
                            )

                            st.write(content)

            # ==========================================
            # FALLBACK
            # ==========================================

            else:

                st.markdown(
                    """
                    <div class="section-title">
                        Analysis result
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(tool_result)

            # ------------------------------------------
            # HUMAN-CONTROL NOTICE
            # ------------------------------------------

            st.markdown(
                (
                    '<div class="info-panel">'
                    '<div class="info-panel-title">'
                    'Evidence for investigation'
                    '</div>'
                    '<p class="info-panel-text">'
                    'SafeSight surfaces historical patterns and related '
                    'reports to support a safety officer&apos;s investigation. '
                    'Results should be interpreted in context before taking '
                    'operational action.'
                    '</p>'
                    '</div>'
                ),
                unsafe_allow_html=True
            )