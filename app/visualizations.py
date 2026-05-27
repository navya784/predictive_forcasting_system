"""Static and interactive visualizations for EDA and forecasting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from app.utils import CHARTS_DIR, DATE_COLUMN, display_name, numeric_columns


PRIMARY_BLUE = "#2563EB"
DARK_NAVY = "#0F172A"
AMBER = "#F59E0B"
RED_ALERT = "#EF4444"
WHITE = "#F8FAFC"
SOFT_GRAY = "#CBD5E1"
BLUE_BAND = "rgba(37,99,235,0.2)"
RED_BAND = "rgba(239,68,68,0.16)"


def _save_plot(path: Path) -> Path:
    """Save the active Matplotlib figure and close it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, facecolor=DARK_NAVY)
    plt.close()
    return path


def generate_eda_charts(df: pd.DataFrame, target_column: str, output_dir: Path | None = None) -> list[Path]:
    """Generate all required EDA charts and save them automatically."""

    output_dir = output_dir or CHARTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    charts: list[Path] = []

    sns.set_theme(style="darkgrid")
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_NAVY,
            "axes.facecolor": DARK_NAVY,
            "axes.edgecolor": SOFT_GRAY,
            "axes.labelcolor": WHITE,
            "xtick.color": SOFT_GRAY,
            "ytick.color": SOFT_GRAY,
            "text.color": WHITE,
        }
    )
    working_df = df.sort_values(DATE_COLUMN).copy()

    plt.figure(figsize=(13, 6))
    sns.lineplot(data=working_df, x=DATE_COLUMN, y=target_column, color=PRIMARY_BLUE, linewidth=2.2)
    plt.title(f"Trend Chart: {display_name(target_column)}")
    plt.xlabel("Date")
    plt.ylabel("Children")
    charts.append(_save_plot(output_dir / "trend_chart.png"))

    plt.figure(figsize=(13, 6))
    plt.plot(working_df[DATE_COLUMN], working_df[target_column], label="Daily", color=PRIMARY_BLUE, alpha=0.55)
    plt.plot(
        working_df[DATE_COLUMN],
        working_df[target_column].rolling(7, min_periods=1).mean(),
        label="7-day rolling mean",
        color=AMBER,
        linewidth=2.2,
    )
    plt.plot(
        working_df[DATE_COLUMN],
        working_df[target_column].rolling(14, min_periods=1).mean(),
        label="14-day rolling mean",
        color=SOFT_GRAY,
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
        sns.heatmap(
            working_df[numeric].corr(),
            annot=True,
            fmt=".2f",
            cmap=sns.color_palette([DARK_NAVY, PRIMARY_BLUE, WHITE, AMBER, RED_ALERT], as_cmap=True),
            center=0,
            linewidths=0.35,
            linecolor=SOFT_GRAY,
        )
        plt.title("Correlation Heatmap")
        charts.append(_save_plot(output_dir / "correlation_heatmap.png"))

    plt.figure(figsize=(10, 5))
    sns.histplot(working_df[target_column], kde=True, color=PRIMARY_BLUE)
    plt.title(f"Distribution Plot: {display_name(target_column)}")
    plt.xlabel("Children")
    charts.append(_save_plot(output_dir / "distribution_plot.png"))

    if len(working_df) >= 30:
        try:
            series = working_df.set_index(DATE_COLUMN)[target_column].asfreq("D")
            decomposition = seasonal_decompose(series, model="additive", period=7)
            figure = decomposition.plot()
            figure.set_size_inches(13, 9)
            figure.suptitle("Seasonal Decomposition", y=1.02, color=WHITE)
            charts.append(_save_plot(output_dir / "seasonal_decomposition.png"))
        except Exception:
            pass

    if "net_pressure" in working_df.columns:
        plt.figure(figsize=(13, 5))
        sns.lineplot(data=working_df, x=DATE_COLUMN, y="net_pressure", color=AMBER, linewidth=2)
        plt.axhline(0, linestyle="--", color=SOFT_GRAY, linewidth=1)
        plt.title("Net Pressure: Transfers - Discharges")
        plt.xlabel("Date")
        plt.ylabel("Children")
        charts.append(_save_plot(output_dir / "net_pressure_chart.png"))

    if "capacity_pressure_index" in working_df.columns:
        plt.figure(figsize=(13, 5))
        sns.lineplot(data=working_df, x=DATE_COLUMN, y="capacity_pressure_index", color=RED_ALERT, linewidth=2)
        plt.axhline(1.0, linestyle="--", color=AMBER, linewidth=1, label="High pressure threshold")
        plt.title("Capacity Pressure Index")
        plt.xlabel("Date")
        plt.ylabel("Index")
        plt.legend()
        charts.append(_save_plot(output_dir / "capacity_pressure_index.png"))

    return charts


def _dark_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    """Apply consistent Plotly dark analytics styling."""

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(color=WHITE, size=18)),
        height=height,
        paper_bgcolor="rgba(15,23,42,0)",
        plot_bgcolor="rgba(15,23,42,0.72)",
        font=dict(color=SOFT_GRAY),
        margin=dict(l=22, r=22, t=58, b=28),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=SOFT_GRAY),
        ),
        xaxis=dict(gridcolor="rgba(203,213,225,0.12)", zerolinecolor="rgba(203,213,225,0.20)"),
        yaxis=dict(gridcolor="rgba(203,213,225,0.12)", zerolinecolor="rgba(203,213,225,0.20)"),
    )
    return fig


def plot_interactive_forecast(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target_column: str,
    surge_threshold: float | None = None,
) -> go.Figure:
    """Create an interactive forecast chart with confidence intervals."""

    recent_history = history_df.tail(150)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent_history[DATE_COLUMN],
            y=recent_history[target_column],
            mode="lines",
            name="Actual values",
            line=dict(color=PRIMARY_BLUE, width=2.7),
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
            fillcolor=BLUE_BAND,
            line=dict(width=0),
            name="Confidence interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df[DATE_COLUMN],
            y=forecast_df["forecast"],
            mode="lines+markers",
            name="Forecast values",
            line=dict(color=AMBER, width=3.1),
            marker=dict(size=6, color=AMBER),
        )
    )

    if surge_threshold is not None:
        max_y = max(float(forecast_df["upper_ci"].max()), surge_threshold * 1.02)
        fig.add_hrect(
            y0=surge_threshold,
            y1=max_y,
            fillcolor=RED_BAND,
            line_width=0,
            annotation_text="Surge zone",
            annotation_position="top left",
        )
        fig.add_hline(
            y=surge_threshold,
            line_color=RED_ALERT,
            line_dash="dash",
            annotation_text="Critical threshold",
        )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Children")
    return _dark_layout(fig, f"Forecast: {display_name(target_column)}", 480)


def plot_interactive_trend(df: pd.DataFrame, target_column: str) -> go.Figure:
    """Create a dark trend chart with rolling average."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[DATE_COLUMN],
            y=df[target_column],
            mode="lines",
            name="Daily value",
            line=dict(color=PRIMARY_BLUE, width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[DATE_COLUMN],
            y=df[target_column].rolling(7, min_periods=1).mean(),
            mode="lines",
            name="7-day average",
            line=dict(color=AMBER, width=2.4, dash="dot"),
        )
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Children")
    return _dark_layout(fig, f"Signal Trend: {display_name(target_column)}", 410)


def plot_net_pressure(df: pd.DataFrame) -> go.Figure | None:
    """Create a dark net-pressure chart if the feature exists."""

    if "net_pressure" not in df.columns:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[DATE_COLUMN],
            y=df["net_pressure"],
            mode="lines",
            name="Net pressure",
            line=dict(color=AMBER, width=2.5),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=SOFT_GRAY)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Transfers - discharges")
    return _dark_layout(fig, "Net Pressure: Transfers - Discharges", 380)


def plot_model_leaderboard(leaderboard: pd.DataFrame, best_model_name: str | None = None) -> go.Figure:
    """Create a dark model-comparison bar chart."""

    chart_df = leaderboard.dropna(subset=["rmse"]).copy()
    colors = [AMBER if model == best_model_name else PRIMARY_BLUE for model in chart_df["model"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_df["model"],
                y=chart_df["rmse"],
                marker=dict(color=colors, line=dict(color=SOFT_GRAY, width=0.5)),
                text=chart_df["rmse"].round(2),
                textposition="outside",
                name="RMSE",
            )
        ]
    )
    fig.update_xaxes(title_text="Model")
    fig.update_yaxes(title_text="RMSE")
    return _dark_layout(fig, "Model Comparison by RMSE", 390)

