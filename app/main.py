"""Premium Streamlit dashboard for care-load forecasting and surge monitoring."""

from __future__ import annotations

import sys
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.forecast import detect_surge_risk, forecast_future, simple_discharge_forecast  # noqa: E402
from app.model_training import train_all_models  # noqa: E402
from app.preprocessing import DataValidationError, available_targets, load_and_preprocess  # noqa: E402
from app.utils import DATE_COLUMN, DEFAULT_DATASET_PATH, DEFAULT_TARGET, DISCHARGE_TARGET, HORIZON_OPTIONS, display_name  # noqa: E402
from app.visualizations import (  # noqa: E402
    plot_interactive_forecast,
    plot_interactive_trend,
    plot_model_leaderboard,
    plot_net_pressure,
)


PRIMARY_BLUE = "#2563EB"
DARK_NAVY = "#0F172A"
AMBER = "#F59E0B"
RED_ALERT = "#EF4444"
WHITE = "#F8FAFC"
SOFT_GRAY = "#CBD5E1"


st.set_page_config(
    page_title="Predictive Forecasting of Care Load & Placement Demand",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Inject a premium dark analytics design system."""

    st.markdown(
        f"""
        <style>
        :root {{
            --blue: {PRIMARY_BLUE};
            --navy: {DARK_NAVY};
            --amber: {AMBER};
            --red: {RED_ALERT};
            --white: {WHITE};
            --gray: {SOFT_GRAY};
        }}

        .stApp {{
            background:
                radial-gradient(circle at 18% 8%, rgba(37,99,235,0.24), transparent 28%),
                radial-gradient(circle at 82% 4%, rgba(245,158,11,0.12), transparent 24%),
                linear-gradient(135deg, #0F172A 0%, #0F172A 56%, #0F172A 100%);
            color: var(--white);
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F172A 0%, #0F172A 100%);
            border-right: 1px solid rgba(203,213,225,0.18);
            box-shadow: 16px 0 40px rgba(15,23,42,0.45);
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--white);
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {{
            color: var(--gray);
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="radio"] {{
            background: rgba(248,250,252,0.06);
            border-color: rgba(203,213,225,0.22);
        }}

        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1480px;
        }}

        @keyframes riseIn {{
            0% {{ opacity: 0; transform: translateY(18px) scale(0.985); }}
            100% {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}

        @keyframes pulseGlow {{
            0%, 100% {{ box-shadow: 0 16px 44px rgba(37,99,235,0.16); }}
            50% {{ box-shadow: 0 18px 54px rgba(37,99,235,0.28); }}
        }}

        .sidebar-logo {{
            padding: 1.05rem 0.95rem;
            margin-bottom: 1rem;
            border-radius: 18px;
            background: rgba(248,250,252,0.07);
            border: 1px solid rgba(203,213,225,0.18);
            box-shadow: inset 0 1px 0 rgba(248,250,252,0.08);
        }}

        .sidebar-logo-title {{
            font-size: 1rem;
            font-weight: 850;
            letter-spacing: 0;
            color: var(--white);
        }}

        .sidebar-logo-subtitle {{
            margin-top: 0.35rem;
            font-size: 0.78rem;
            color: var(--gray);
        }}

        .nav-pill {{
            padding: 0.62rem 0.75rem;
            margin: 0.35rem 0;
            border-radius: 14px;
            background: rgba(248,250,252,0.04);
            border: 1px solid rgba(203,213,225,0.12);
            color: var(--gray);
            transition: all 180ms ease;
        }}

        .nav-pill:hover {{
            background: rgba(37,99,235,0.20);
            border-color: rgba(37,99,235,0.58);
            color: var(--white);
            transform: translateX(2px);
        }}

        .ai-hero {{
            border-radius: 24px;
            padding: 1.45rem 1.55rem;
            background:
                linear-gradient(135deg, rgba(37,99,235,0.28), rgba(15,23,42,0.68)),
                rgba(248,250,252,0.06);
            border: 1px solid rgba(203,213,225,0.18);
            box-shadow: 0 24px 70px rgba(15,23,42,0.55);
            backdrop-filter: blur(18px);
            animation: riseIn 560ms cubic-bezier(0.22,1,0.36,1);
        }}

        .hero-eyebrow {{
            color: var(--amber);
            font-weight: 760;
            font-size: 0.78rem;
            letter-spacing: 0;
            text-transform: uppercase;
        }}

        .hero-title {{
            margin: 0.35rem 0 0;
            color: var(--white);
            font-size: clamp(1.8rem, 3.5vw, 3.25rem);
            line-height: 1.06;
            font-weight: 900;
            letter-spacing: 0;
        }}

        .hero-subtitle {{
            max-width: 920px;
            margin-top: 0.8rem;
            color: var(--gray);
            font-size: 1rem;
            line-height: 1.55;
        }}

        .last-updated {{
            margin-top: 1rem;
            color: var(--gray);
            font-size: 0.86rem;
        }}

        .glass-panel {{
            border-radius: 22px;
            padding: 1rem;
            background: rgba(248,250,252,0.06);
            border: 1px solid rgba(203,213,225,0.16);
            box-shadow: 0 18px 54px rgba(15,23,42,0.48);
            backdrop-filter: blur(18px);
            animation: riseIn 580ms cubic-bezier(0.22,1,0.36,1);
        }}

        .section-title {{
            color: var(--white);
            font-size: 1.05rem;
            font-weight: 850;
            margin: 1.15rem 0 0.65rem;
            letter-spacing: 0;
        }}

        .kpi-card {{
            min-height: 154px;
            border-radius: 22px;
            padding: 1rem;
            background: rgba(248,250,252,0.07);
            border: 1px solid rgba(203,213,225,0.15);
            box-shadow: 0 18px 48px rgba(15,23,42,0.48);
            backdrop-filter: blur(16px);
            animation: riseIn 640ms cubic-bezier(0.22,1,0.36,1);
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(37,99,235,0.65);
            animation: pulseGlow 1500ms ease-in-out infinite;
        }}

        .kpi-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 14px;
            font-weight: 850;
            margin-bottom: 0.8rem;
        }}

        .kpi-label {{
            color: var(--gray);
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }}

        .kpi-value {{
            color: var(--white);
            font-size: clamp(1.45rem, 2.2vw, 2.2rem);
            line-height: 1;
            font-weight: 900;
            letter-spacing: 0;
        }}

        .kpi-trend {{
            color: var(--gray);
            margin-top: 0.7rem;
            font-size: 0.78rem;
        }}

        .tone-blue {{ color: var(--blue); background: rgba(37,99,235,0.16); }}
        .tone-amber {{ color: var(--amber); background: rgba(245,158,11,0.16); }}
        .tone-red {{ color: var(--red); background: rgba(239,68,68,0.16); }}
        .tone-stable {{ color: var(--white); background: rgba(203,213,225,0.14); }}

        .risk-alert {{
            border-radius: 22px;
            padding: 1rem 1.1rem;
            color: var(--white);
            border: 1px solid rgba(203,213,225,0.16);
            box-shadow: 0 18px 50px rgba(15,23,42,0.48);
            animation: riseIn 520ms cubic-bezier(0.22,1,0.36,1);
        }}

        .risk-low {{ background: linear-gradient(135deg, rgba(37,99,235,0.32), rgba(15,23,42,0.68)); }}
        .risk-medium {{ background: linear-gradient(135deg, rgba(245,158,11,0.34), rgba(15,23,42,0.70)); }}
        .risk-high {{ background: linear-gradient(135deg, rgba(239,68,68,0.44), rgba(15,23,42,0.72)); }}

        .risk-label {{
            font-size: 0.78rem;
            color: var(--gray);
            text-transform: uppercase;
            font-weight: 800;
        }}

        .risk-value {{
            margin-top: 0.25rem;
            font-size: clamp(1.25rem, 2vw, 2rem);
            font-weight: 900;
            color: var(--white);
        }}

        .stButton > button,
        .stDownloadButton > button {{
            border-radius: 16px;
            border: 1px solid rgba(37,99,235,0.65);
            background: linear-gradient(135deg, #2563EB, #0F172A);
            color: var(--white);
            font-weight: 850;
            min-height: 46px;
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 14px 34px rgba(37,99,235,0.28);
            border-color: var(--amber);
        }}

        div[data-testid="stMetric"] {{
            border-radius: 18px;
            background: rgba(248,250,252,0.06);
            border: 1px solid rgba(203,213,225,0.13);
            padding: 0.85rem;
            color: var(--white);
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(203,213,225,0.14);
        }}

        .stAlert {{
            border-radius: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_motion_header() -> None:
    """Render the AI-powered header with st.iframe instead of deprecated components."""

    updated = datetime.now().strftime("%d %b %Y, %I:%M %p")
    header_html = f"""
    <div class="ai-hero">
        <div class="hero-eyebrow">AI Healthcare Intelligence Platform</div>
        <div class="hero-title">Predictive Forecasting of Care Load & Placement Demand</div>
        <div class="hero-subtitle">
            Executive monitoring for HHS care load, discharge demand, placement pressure,
            model performance, and early surge-risk detection.
        </div>
        <div class="last-updated">Last updated: {updated}</div>
    </div>
    <style>
        body {{ margin:0; background:transparent; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
        @keyframes riseIn {{ from {{ opacity:0; transform:translateY(18px); }} to {{ opacity:1; transform:translateY(0); }} }}
        .ai-hero {{
            border-radius:24px; padding:24px; color:#F8FAFC;
            background:linear-gradient(135deg, rgba(37,99,235,0.32), rgba(15,23,42,0.78));
            border:1px solid rgba(203,213,225,0.18); box-shadow:0 24px 70px rgba(0,0,0,0.32);
            animation:riseIn 560ms cubic-bezier(0.22,1,0.36,1);
        }}
        .hero-eyebrow {{ color:#F59E0B; font-size:13px; font-weight:800; text-transform:uppercase; }}
        .hero-title {{ margin-top:8px; font-size:clamp(28px,4vw,48px); line-height:1.06; font-weight:900; letter-spacing:0; }}
        .hero-subtitle {{ max-width:920px; margin-top:12px; color:#CBD5E1; font-size:16px; line-height:1.55; }}
        .last-updated {{ margin-top:16px; color:#CBD5E1; font-size:13px; }}
    </style>
    """
    st.iframe(header_html, height=210, width="stretch")


def render_sidebar() -> tuple[bytes | None, int, str, str, bool]:
    """Render sidebar controls."""

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="sidebar-logo-title">HHS AI Monitor</div>
                <div class="sidebar-logo-subtitle">Care load forecasting command center</div>
            </div>
            <div class="nav-pill">[AI] Dashboard</div>
            <div class="nav-pill">[DATA] Dataset Intake</div>
            <div class="nav-pill">[MODEL] Forecast Models</div>
            <div class="nav-pill">[RISK] Risk Alerts</div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Dataset upload", type=["csv"])
        horizon = st.radio("Forecast horizon", HORIZON_OPTIONS, index=2, horizontal=True)
        model_choice = st.selectbox(
            "Model selection",
            [
                "Auto Best",
                "Naive Forecast",
                "Moving Average",
                "ARIMA",
                "SARIMA",
                "Exponential Smoothing",
                "Random Forest",
                "Gradient Boosting",
            ],
        )
        run_forecast = st.button("Run AI Forecast", width="stretch")

    uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None
    return uploaded_bytes, horizon, model_choice, "", run_forecast


def kpi_card(label: str, value: str, icon: str, tone: str, trend: str) -> None:
    """Render one animated KPI card."""

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon {tone}">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-trend">{trend}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_banner(risk_level: str, risk_percent: float, threshold: float) -> None:
    """Render the risk alert panel."""

    if "HIGH" in risk_level:
        css_class = "risk-high"
        narrative = "Critical surge zone active"
    elif "MODERATE" in risk_level:
        css_class = "risk-medium"
        narrative = "Watchlist threshold approaching"
    else:
        css_class = "risk-low"
        narrative = "Capacity profile currently stable"

    st.markdown(
        f"""
        <div class="risk-alert {css_class}">
            <div class="risk-label">Risk Alert Panel</div>
            <div class="risk-value">{risk_level}</div>
            <div style="margin-top:0.45rem;color:{SOFT_GRAY};">
                {narrative} | Breach probability: {risk_percent:.1f}% | Threshold: {threshold:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dashboard_data(uploaded_bytes: bytes | None) -> tuple[pd.DataFrame, dict]:
    """Load and preprocess dashboard data."""

    if uploaded_bytes:
        return load_and_preprocess(BytesIO(uploaded_bytes))
    return load_and_preprocess(DEFAULT_DATASET_PATH)


@st.cache_resource(show_spinner=False)
def train_dashboard_models(df_json: str, target_column: str) -> dict:
    """Train models with Streamlit caching."""

    df = pd.read_json(StringIO(df_json), orient="split")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return train_all_models(df, target_column)


def render_model_table(leaderboard: pd.DataFrame, best_model_name: str) -> None:
    """Render model comparison with highlighted best model and progress bars."""

    table = leaderboard.copy()
    table.insert(0, "rank", range(1, len(table) + 1))
    table.insert(1, "status", table["model"].apply(lambda value: "BEST" if value == best_model_name else ""))
    for column in ["accuracy_percent", "mape", "r2_score"]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0)

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "accuracy_percent": st.column_config.ProgressColumn(
                "Accuracy %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "mape": st.column_config.ProgressColumn(
                "MAPE",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
            "r2_score": st.column_config.ProgressColumn(
                "R2 Score",
                min_value=-1,
                max_value=1,
                format="%.2f",
            ),
        },
    )


def main() -> None:
    """Run the Streamlit dashboard."""

    inject_css()
    uploaded_bytes, horizon, model_choice, _, run_forecast = render_sidebar()
    render_motion_header()

    try:
        df, metadata = load_dashboard_data(uploaded_bytes)
        targets = available_targets(df)
    except DataValidationError as exc:
        st.error(f"Dataset validation issue: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unable to load dataset. Please check the CSV format. Details: {exc}")
        st.stop()

    target_default = targets.index(DEFAULT_TARGET) if DEFAULT_TARGET in targets else 0
    target_column = st.selectbox(
        "Forecast target",
        targets,
        index=target_default,
        format_func=display_name,
        width="stretch",
    )

    st.markdown('<div class="section-title">Mission Dataset Overview</div>', unsafe_allow_html=True)
    overview_cols = st.columns(4)
    overview_cols[0].metric("Rows", f"{len(df):,}")
    overview_cols[1].metric("Signals", f"{df.shape[1]:,}")
    overview_cols[2].metric("Start", str(df[DATE_COLUMN].min().date()))
    overview_cols[3].metric("End", str(df[DATE_COLUMN].max().date()))

    with st.expander("Preprocessing intelligence summary", expanded=False):
        st.json(metadata)

    st.markdown('<div class="section-title">Live Signal Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_interactive_trend(df, target_column), width="stretch")

    pressure_chart = plot_net_pressure(df)
    if pressure_chart is not None:
        st.plotly_chart(pressure_chart, width="stretch")

    if run_forecast:
        progress = st.progress(0)
        try:
            with st.spinner("Initializing AI forecasting engine..."):
                progress.progress(15)
                df_json = df.to_json(orient="split", date_format="iso")
                progress.progress(35)
                results = train_dashboard_models(df_json, target_column)
                progress.progress(65)
                model_name = results["best_model_name"] if model_choice == "Auto Best" else model_choice

                if model_name not in results["artifacts"]:
                    st.error(f"{model_name} could not be trained for this dataset. Try Auto Best.")
                    st.stop()

                artifact = results["artifacts"][model_name]
                forecast_df = forecast_future(artifact, df, horizon=horizon)
                risk = detect_surge_risk(forecast_df, df, target_column)
                progress.progress(100)

            st.success("Forecast package generated successfully.")

            st.markdown('<div class="section-title">Risk Alert Panel</div>', unsafe_allow_html=True)
            risk_banner(
                str(risk["risk_level"]),
                float(risk["capacity_breach_risk_percent"]),
                float(risk["capacity_threshold"]),
            )

            st.markdown('<div class="section-title">Executive KPI Cards</div>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi_card(
                    "Forecast Accuracy",
                    f"{artifact.metrics.get('accuracy_percent', 0):.1f}%",
                    "AI",
                    "tone-blue",
                    "Model confidence score",
                )
            with k2:
                kpi_card(
                    "Surge Warning",
                    f"{risk['surge_lead_time_days']} days",
                    "SW",
                    "tone-amber",
                    "Lead time to watch threshold",
                )
            with k3:
                kpi_card(
                    "Capacity Risk",
                    f"{risk['capacity_breach_risk_percent']:.1f}%",
                    "CR",
                    "tone-red",
                    "Projected breach exposure",
                )
            with k4:
                kpi_card(
                    "Stable Capacity",
                    f"{risk['forecast_stability_index']:.1f}",
                    "SC",
                    "tone-stable",
                    "Forecast stability index",
                )

            st.markdown('<div class="section-title">Forecast Chart</div>', unsafe_allow_html=True)
            st.plotly_chart(
                plot_interactive_forecast(
                    df,
                    forecast_df,
                    target_column,
                    surge_threshold=float(risk["capacity_threshold"]),
                ),
                width="stretch",
            )

            if DISCHARGE_TARGET in df.columns:
                st.markdown('<div class="section-title">Discharge Prediction</div>', unsafe_allow_html=True)
                discharge_forecast = simple_discharge_forecast(df, DISCHARGE_TARGET, horizon)
                st.plotly_chart(
                    plot_interactive_forecast(df, discharge_forecast, DISCHARGE_TARGET),
                    width="stretch",
                )

            st.markdown('<div class="section-title">Model Comparison Panel</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_model_leaderboard(results["leaderboard"], results["best_model_name"]), width="stretch")
            render_model_table(results["leaderboard"], results["best_model_name"])

            st.download_button(
                label="Download Forecast CSV",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name="forecast_results.csv",
                mime="text/csv",
                width="stretch",
            )
        except Exception as exc:
            st.error(f"Forecasting failed gracefully: {exc}")


if __name__ == "__main__":
    main()
