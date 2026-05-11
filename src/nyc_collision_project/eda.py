from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd

from .config import ProjectConfig
from .features import TARGET_COL
from .io_utils import write_table


plt.switch_backend("Agg")


def _savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _safe_plot(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            warnings.warn(f"Skipping plot {func.__name__}: {e}")
            return None
    return wrapper


@_safe_plot
def plot_monthly_trend(config: ProjectConfig, df: pd.DataFrame):
    if "crash_date" not in df.columns:
        return
    d = df.dropna(subset=["crash_date"]).copy()
    monthly = d.set_index("crash_date").resample("MS").size()
    plt.figure(figsize=(8, 4))
    monthly.plot()
    plt.title("NYC collisions by month")
    plt.xlabel("Month")
    plt.ylabel("Collision count")
    _savefig(config.outputs_dir / "figures" / "fig_monthly_trend.png")
    write_table(monthly.reset_index(name="collision_count"), config.outputs_dir / "tables" / "fig_monthly_trend_data.csv")


@_safe_plot
def plot_injury_rate_by_year(config: ProjectConfig, df: pd.DataFrame):
    if "crash_year" not in df.columns:
        return
    y = df.groupby("crash_year", dropna=True)[TARGET_COL].agg(["count", "mean"]).reset_index()
    y.columns = ["crash_year", "collisions", "injury_or_fatality_rate"]
    plt.figure(figsize=(7, 4))
    plt.plot(y["crash_year"], y["injury_or_fatality_rate"], marker="o")
    plt.title("Injury-producing collision rate by year")
    plt.xlabel("Year")
    plt.ylabel("Rate")
    _savefig(config.outputs_dir / "figures" / "fig_injury_rate_by_year.png")
    write_table(y, config.outputs_dir / "tables" / "fig_injury_rate_by_year_data.csv")


@_safe_plot
def plot_borough_hour_heatmap(config: ProjectConfig, df: pd.DataFrame):
    if "borough" not in df.columns or "crash_hour" not in df.columns:
        return
    pivot = pd.crosstab(df["borough"].fillna("unknown"), df["crash_hour"].fillna(-1))
    plt.figure(figsize=(9, 4.8))
    plt.imshow(pivot.values, aspect="auto")
    plt.title("Collision count heatmap by borough and hour")
    plt.xlabel("Hour")
    plt.ylabel("Borough")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    plt.colorbar(label="Count")
    _savefig(config.outputs_dir / "figures" / "fig_borough_hour_heatmap.png")
    write_table(pivot.reset_index(), config.outputs_dir / "tables" / "fig_borough_hour_heatmap_data.csv")


@_safe_plot
def plot_top_factors(config: ProjectConfig, df: pd.DataFrame):
    factor_cols = [c for c in df.columns if c.startswith("contributing_factor_vehicle")]
    if not factor_cols:
        return
    vals = pd.concat([df[c].astype("string") for c in factor_cols], ignore_index=True)
    vals = vals.dropna().str.lower().str.strip()
    vals = vals[~vals.isin(["", "unspecified", "unknown", "nan"])]
    top = vals.value_counts().head(15).sort_values()
    plt.figure(figsize=(8, 5))
    top.plot(kind="barh")
    plt.title("Top recorded contributing factors")
    plt.xlabel("Count across factor fields")
    _savefig(config.outputs_dir / "figures" / "fig_top_factors.png")
    write_table(top.reset_index().rename(columns={"index": "factor", 0: "count"}), config.outputs_dir / "tables" / "fig_top_factors_data.csv")


@_safe_plot
def plot_injury_by_weekday_hour(config: ProjectConfig, df: pd.DataFrame):
    if "crash_dayofweek" not in df.columns or "crash_hour" not in df.columns:
        return
    rate = df.pivot_table(values=TARGET_COL, index="crash_dayofweek", columns="crash_hour", aggfunc="mean")
    plt.figure(figsize=(9, 4.5))
    plt.imshow(rate.values, aspect="auto", vmin=0)
    plt.title("Injury/fatality rate by weekday and hour")
    plt.xlabel("Hour")
    plt.ylabel("Day of week, Monday=0")
    plt.xticks(range(len(rate.columns)), rate.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(rate.index)), rate.index, fontsize=8)
    plt.colorbar(label="Rate")
    _savefig(config.outputs_dir / "figures" / "fig_injury_by_weekday_hour.png")
    write_table(rate.reset_index(), config.outputs_dir / "tables" / "fig_injury_by_weekday_hour_data.csv")


@_safe_plot
def plot_vehicle_person_injury_rates(config: ProjectConfig, df: pd.DataFrame):
    rows = []
    for col in ["vehicle_records_count", "person_records_count", "person_pedestrian_count", "person_bicyclist_count"]:
        if col in df.columns:
            temp = df[[col, TARGET_COL]].copy()
            temp[col] = pd.to_numeric(temp[col], errors="coerce").clip(upper=10)
            g = temp.groupby(col)[TARGET_COL].agg(["count", "mean"]).reset_index()
            g["feature"] = col
            rows.append(g.rename(columns={col: "bucket", "mean": "injury_or_fatality_rate"}))
    if not rows:
        return
    out = pd.concat(rows, ignore_index=True)
    write_table(out, config.outputs_dir / "tables" / "fig_injury_rate_by_involvement_data.csv")
    first = out[out["feature"] == out["feature"].iloc[0]]
    plt.figure(figsize=(7, 4))
    plt.plot(first["bucket"], first["injury_or_fatality_rate"], marker="o")
    plt.title(f"Injury/fatality rate by {first['feature'].iloc[0]}")
    plt.xlabel("Count bucket")
    plt.ylabel("Rate")
    _savefig(config.outputs_dir / "figures" / "fig_injury_rate_by_involvement.png")


@_safe_plot
def plot_missingness(config: ProjectConfig, df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False).head(30)
    plt.figure(figsize=(8, 6))
    miss.sort_values().plot(kind="barh")
    plt.title("Top missingness rates after feature construction")
    plt.xlabel("Missing rate")
    _savefig(config.outputs_dir / "figures" / "fig_missingness_summary.png")
    write_table(miss.reset_index().rename(columns={"index": "column", 0: "missing_rate"}), config.outputs_dir / "tables" / "fig_missingness_summary_data.csv")


def run_eda(config: ProjectConfig, df: pd.DataFrame) -> None:
    plot_monthly_trend(config, df)
    plot_injury_rate_by_year(config, df)
    plot_borough_hour_heatmap(config, df)
    plot_top_factors(config, df)
    plot_injury_by_weekday_hour(config, df)
    plot_vehicle_person_injury_rates(config, df)
    plot_missingness(config, df)
