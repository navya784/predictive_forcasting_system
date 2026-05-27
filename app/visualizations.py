"""Static and interactive visualizations for EDA and forecasting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from app.utils import CHARTS_DIR, DATE_COLUMN, display_name, numeric_columns


def _save_plot(path: Path) -> Path:
    """Save the active Matplotlib figure and close it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def generate_eda_charts(df: pd.DataFrame, target_column: str, output_dir: Path | None = None) -> list[Path]:
    """Generate all required EDA charts and save them automatically."""

    output_dir = output_dir or CHARTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    charts: list[Path] = []

    sns.set_theme(style="whitegrid")
    working_df = df.sort_values(DATE_COLUMN).copy()

    plt.figure(figsize=(13, 6))
    sns.lineplot(data=working_df, x=DATE_COLUMN, y=target_column, color="#1d4ed8", linewidth=2.2)
    plt.title(f"Trend Chart: {display_name(target_column)}")
    plt.xlabel("Date")
    plt.ylabel("Children")
    charts.append(_save_plot(output_dir / "trend_chart.png"))

    plt.figure(figsize=(13, 6))
    plt.plot(working_df[DATE_COLUMN], working_df[target_column], label="Daily", alpha=0.55)
    plt.plot(
        working_df[DATE_COLUMN],
        working_df[target_column].rolling(7, min_periods=1).mean(),
        label="7-day rolling mean",
        linewidth=2.2,
    )
    plt.plot(
        working_df[DATE_COLUMN],
        working_df[target_column].rolling(14, min_periods=1).mean(),
        label="14-day rolling mean",
        linewidth=2.2,
    )
    plt.title("Rolling Statistics")
    plt.xlabel("Date")
    plt.ylabel("Children")
    plt.legend()
    charts.append(_save_plot(output_dir / "rolling_statistics.png"))

    numeric = numeric_columns(working_df, exclude=[DATE_COLUMN])
    if len(numeric) >= 2:
        plt.figure(figsize=(11, 8))
        sns.heatmap(working_df[numeric].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title("Correlation Heatmap")
        charts.append(_save_plot(output_dir / "correlation_heatmap.png"))

    plt.figure(figsize=(10, 5))
    sns.histplot(working_df[target_column], kde=True, color="#0f766e")
    plt.title(f"Distribution Plot: {display_name(target_column)}")
    plt.xlabel("Children")
    charts.append(_save_plot(output_dir / "distribution_plot.png"))

    if len(working_df) >= 30:
        try:
            series = working_df.set_index(DATE_COLUMN)[target_column].asfreq("D")
            decomposition = seasonal_decompose(series, model="additive", period=7)
            figure = decomposition.plot()
            figure.set_size_inches(13, 9)
            figure.suptitle("Seasonal Decomposition", y=1.02)
            charts.append(_save_plot(output_dir / "seasonal_decomposition.png"))
        except Exception:
            # Seasonal decomposition can fail on unusual data. Other EDA charts
            # still provide useful diagnostics.
            pass

    if "net_pressure" in working_df.columns:
        plt.figure(figsize=(13, 5))
        sns.lineplot(data=working_df, x=DATE_COLUMN, y="net_pressure", color="#b45309", linewidth=2)
        plt.axhline(0, linestyle="--", color="#111827", linewidth=1)
        plt.title("Net Pressure: Transfers - Discharges")
        plt.xlabel("Date")
        plt.ylabel("Children")
        charts.append(_save_plot(output_dir / "net_pressure_chart.png"))

    if "capacity_pressure_index" in working_df.columns:
        plt.figure(figsize=(13, 5))
        sns.lineplot(data=working_df, x=DATE_COLUMN, y="capacity_pressure_index", color="#dc2626", linewidth=2)
        plt.axhline(1.0, linestyle="--", color="#111827", linewidth=1, label="High pressure threshold")
        plt.title("Capacity Pressure Index")
        plt.xlabel("Date")
        plt.ylabel("Index")
        plt.legend()
        charts.append(_save_plot(output_dir / "capacity_pressure_index.png"))

    return charts


def plot_interactive_forecast(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target_column: str,
) -> go.Figure:
    """Create an interactive forecast chart with confidence intervals."""

    recent_history = history_df.tail(150)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent_history[DATE_COLUMN],
            y=recent_history[target_column],
            mode="lines",
            name="Historical",
            line=dict(color="#1d4ed8", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df[DATE_COLUMN],
            y=forecast_df["upper_ci"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df[DATE_COLUMN],
            y=forecast_df["lower_ci"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(29,78,216,0.16)",
            line=dict(width=0),
            name="95% confidence interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df[DATE_COLUMN],
            y=forecast_df["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#f97316", width=3),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Forecast: {display_name(target_column)}",
        xaxis_title="Date",
        yaxis_title="Children",
        height=460,
        margin=dict(l=20, r=20, t=48, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_interactive_trend(df: pd.DataFrame, target_column: str) -> go.Figure:
    """Create a trend chart with rolling average."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df[DATE_COLUMN], y=df[target_column], mode="lines", name="Daily value")
    )
    fig.add_trace(
        go.Scatter(
            x=df[DATE_COLUMN],
            y=df[target_column].rolling(7, min_periods=1).mean(),
            mode="lines",
            name="7-day average",
            line=dict(dash="dot", width=2.2),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Trend: {display_name(target_column)}",
        xaxis_title="Date",
        yaxis_title="Children",
        height=390,
        margin=dict(l=20, r=20, t=48, b=24),
    )
    return fig


def plot_net_pressure(df: pd.DataFrame) -> go.Figure | None:
    """Create a net pressure chart if the feature exists."""

    if "net_pressure" not in df.columns:
        return None

    fig = px.line(
        df,
        x=DATE_COLUMN,
        y="net_pressure",
        title="Net Pressure: Transfers - Discharges",
        template="plotly_white",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#111827")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=48, b=24))
    return fig


def plot_model_leaderboard(leaderboard: pd.DataFrame) -> go.Figure:
    """Create a compact model-comparison bar chart."""

    chart_df = leaderboard.dropna(subset=["rmse"]).copy()
    fig = px.bar(
        chart_df,
        x="model",
        y="rmse",
        color="model",
        title="Model Comparison by RMSE",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=48, b=90))
    return fig

