"""Professional Streamlit dashboard for the forecasting project."""

from __future__ import annotations

import sys
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

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


st.set_page_config(
    page_title="Predictive Forecasting of Care Load & Placement Demand",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


COLOR_THEMES = {
    "Federal Blue": {
        "primary": "#1d4ed8",
        "secondary": "#0f766e",
        "accent": "#f97316",
        "background": "#f8fafc",
        "text": "#111827",
    },
    "Civic Teal": {
        "primary": "#0f766e",
        "secondary": "#7c3aed",
        "accent": "#ca8a04",
        "background": "#fbfdf8",
        "text": "#172033",
    },
    "Executive Indigo": {
        "primary": "#4338ca",
        "secondary": "#0891b2",
        "accent": "#e11d48",
        "background": "#f9fafb",
        "text": "#101828",
    },
}


def inject_css(theme: dict[str, str]) -> None:
    """Apply a clean government-style theme."""

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(120deg, rgba(29,78,216,0.07), rgba(15,118,110,0.06)), {theme["background"]};
            color: {theme["text"]};
        }}
        section[data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.96);
            border-right: 1px solid rgba(17,24,39,0.08);
        }}
        @keyframes panelEnter {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .kpi-card {{
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(17,24,39,0.09);
            border-radius: 8px;
            padding: 1rem;
            min-height: 118px;
            box-shadow: 0 10px 26px rgba(17,24,39,0.07);
            animation: panelEnter 520ms cubic-bezier(0.22,1,0.36,1);
        }}
        .kpi-label {{
            font-size: 0.86rem;
            color: rgba(17,24,39,0.66);
            margin-bottom: 0.45rem;
        }}
        .kpi-value {{
            font-size: clamp(1.35rem, 2vw, 2rem);
            font-weight: 760;
            letter-spacing: 0;
            overflow-wrap: anywhere;
            color: {theme["text"]};
        }}
        .risk-high {{
            border-left: 6px solid #dc2626;
            background: #fff1f2;
            color: #991b1b;
            padding: 1rem;
            border-radius: 8px;
            font-weight: 750;
        }}
        .risk-medium {{
            border-left: 6px solid #f97316;
            background: #fff7ed;
            color: #9a3412;
            padding: 1rem;
            border-radius: 8px;
            font-weight: 750;
        }}
        .risk-low {{
            border-left: 6px solid #0f766e;
            background: #ecfdf5;
            color: #065f46;
            padding: 1rem;
            border-radius: 8px;
            font-weight: 750;
        }}
        .section-title {{
            margin: 1rem 0 0.55rem;
            font-size: 1.1rem;
            font-weight: 780;
        }}
        .stButton > button,
        .stDownloadButton > button {{
            border-radius: 8px;
            border: 0;
            background: linear-gradient(135deg, {theme["primary"]}, {theme["secondary"]});
            color: white;
            font-weight: 760;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_motion_hero(theme: dict[str, str]) -> None:
    """Render a motion-enhanced header.

    Framer Motion is loaded from a CDN when available. If the browser is offline,
    the CSS animation still provides a smooth fallback.
    """

    components.html(
        f"""
        <div id="hero" class="hero">
          <h1>Predictive Forecasting of Care Load & Placement Demand</h1>
          <p>Forecast HHS care load, discharge demand, intake pressure, and surge risk with explainable models.</p>
        </div>
        <script type="module">
          try {{
            const motion = await import("https://esm.sh/framer-motion@11.3.19?bundle");
            if (motion && motion.animate) {{
              motion.animate("#hero", {{ opacity: [0, 1], y: [16, 0] }}, {{ duration: 0.55 }});
            }}
          }} catch (error) {{}}
        </script>
        <style>
          body {{ margin: 0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
          .hero {{
            border-radius: 8px;
            padding: 20px 22px;
            color: white;
            background: linear-gradient(135deg, {theme["primary"]}, {theme["secondary"]});
            box-shadow: 0 16px 40px rgba(17,24,39,0.16);
          }}
          .hero h1 {{
            margin: 0;
            font-size: clamp(24px, 4vw, 38px);
            line-height: 1.14;
            letter-spacing: 0;
          }}
          .hero p {{
            margin: 10px 0 0;
            opacity: 0.94;
            font-size: 16px;
          }}
        </style>
        """,
        height=150,
    )


def kpi_card(label: str, value: str) -> None:
    """Render one KPI card."""

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
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
    """Train models with caching to keep the dashboard responsive."""

    df = pd.read_json(StringIO(df_json), orient="split")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return train_all_models(df, target_column)


def render_risk_banner(risk_level: str) -> None:
    """Show a capacity risk banner."""

    if "HIGH" in risk_level:
        css_class = "risk-high"
    elif "MODERATE" in risk_level:
        css_class = "risk-medium"
    else:
        css_class = "risk-low"

    st.markdown(f'<div class="{css_class}">{risk_level}</div>', unsafe_allow_html=True)


def main() -> None:
    """Run the Streamlit dashboard."""

    with st.sidebar:
        st.header("Dashboard Controls")
        theme_name = st.selectbox("Color theme", list(COLOR_THEMES.keys()), index=0)

    theme = COLOR_THEMES[theme_name]
    inject_css(theme)
    render_motion_hero(theme)

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
        horizon = st.radio("Forecast horizon", HORIZON_OPTIONS, index=2, horizontal=True)

    try:
        uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None
        df, metadata = load_dashboard_data(uploaded_bytes)
        targets = available_targets(df)
    except DataValidationError as exc:
        st.error(f"Dataset validation issue: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unable to load dataset. Please check the CSV format. Details: {exc}")
        st.stop()

    default_index = targets.index(DEFAULT_TARGET) if DEFAULT_TARGET in targets else 0
    with st.sidebar:
        target_column = st.selectbox("Forecast target", targets, index=default_index, format_func=display_name)
        model_choice = st.selectbox(
            "Select model",
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
        run_forecast = st.button("Run Forecast", use_container_width=True)

    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Start Date", str(df[DATE_COLUMN].min().date()))
    c4.metric("End Date", str(df[DATE_COLUMN].max().date()))

    with st.expander("Preprocessing Summary", expanded=False):
        st.json(metadata)

    st.markdown('<div class="section-title">Exploratory Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_interactive_trend(df, target_column), use_container_width=True)

    pressure_chart = plot_net_pressure(df)
    if pressure_chart is not None:
        st.plotly_chart(pressure_chart, use_container_width=True)

    if run_forecast:
        try:
            with st.spinner("Training models and generating forecasts..."):
                df_json = df.to_json(orient="split", date_format="iso")
                results = train_dashboard_models(df_json, target_column)
                model_name = results["best_model_name"] if model_choice == "Auto Best" else model_choice

                if model_name not in results["artifacts"]:
                    st.error(f"{model_name} could not be trained for this dataset. Try Auto Best.")
                    st.stop()

                artifact = results["artifacts"][model_name]
                forecast_df = forecast_future(artifact, df, horizon=horizon)
                risk = detect_surge_risk(forecast_df, df, target_column)

            st.markdown('<div class="section-title">Capacity Risk Indicator</div>', unsafe_allow_html=True)
            render_risk_banner(str(risk["risk_level"]))

            st.markdown('<div class="section-title">KPI Cards</div>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi_card("Forecast Accuracy", f"{artifact.metrics.get('accuracy_percent', 0):.1f}%")
            with k2:
                kpi_card("Surge Lead Time", f"{risk['surge_lead_time_days']} days")
            with k3:
                kpi_card("Capacity Breach Risk", f"{risk['capacity_breach_risk_percent']:.1f}%")
            with k4:
                kpi_card("Forecast Stability", f"{risk['forecast_stability_index']:.1f}")

            st.markdown('<div class="section-title">Forecast Chart and Confidence Intervals</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_interactive_forecast(df, forecast_df, target_column), use_container_width=True)

            if DISCHARGE_TARGET in df.columns:
                st.markdown('<div class="section-title">Discharge Prediction</div>', unsafe_allow_html=True)
                discharge_forecast = simple_discharge_forecast(df, DISCHARGE_TARGET, horizon)
                st.plotly_chart(plot_interactive_forecast(df, discharge_forecast, DISCHARGE_TARGET), use_container_width=True)

            st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_model_leaderboard(results["leaderboard"]), use_container_width=True)
            st.dataframe(results["leaderboard"], use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Download Forecast</div>', unsafe_allow_html=True)
            st.download_button(
                label="Download Forecast CSV",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name="forecast_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Forecasting failed gracefully: {exc}")


if __name__ == "__main__":
    main()
