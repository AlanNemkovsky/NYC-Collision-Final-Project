from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .io_utils import write_parquet_or_pickle, write_table, write_text


TARGET_COL = "injury_or_fatality"
ID_COL = "collision_id"


DIRECT_TARGET_OR_LEAKAGE_COLS = {
    "number_of_persons_injured", "number_of_persons_killed",
    "number_of_pedestrians_injured", "number_of_pedestrians_killed",
    "number_of_cyclist_injured", "number_of_cyclist_killed",
    "number_of_motorist_injured", "number_of_motorist_killed",
    "persons_injured", "persons_killed", "pedestrians_injured", "pedestrians_killed",
    "cyclist_injured", "cyclist_killed", "motorist_injured", "motorist_killed",
    "total_injury_fatality_count",
}


RAW_TEXT_EXCLUDE = {
    "crash_time", "location", "on_street_name", "cross_street_name", "off_street_name",
}


def _season_from_month(month: pd.Series) -> pd.Series:
    month_num = pd.to_numeric(month, errors="coerce")
    out = pd.Series("unknown", index=month.index, dtype="object")
    out[month_num.isin([12, 1, 2])] = "winter"
    out[month_num.isin([3, 4, 5])] = "spring"
    out[month_num.isin([6, 7, 8])] = "summer"
    out[month_num.isin([9, 10, 11])] = "fall"
    return out


def _collapse_rare(s: pd.Series, top_n: int) -> pd.Series:
    x = s.astype("string").str.lower().str.strip().fillna("unknown")
    x = x.replace({"": "unknown", "unspecified": "unspecified", "nan": "unknown"})
    top = set(x.value_counts(dropna=False).head(top_n).index.astype(str))
    return x.where(x.isin(top), other="other")


def build_feature_matrix(config: ProjectConfig, crashes: pd.DataFrame, vehicles_agg: pd.DataFrame, persons_agg: pd.DataFrame) -> pd.DataFrame:
    df = crashes.copy()

    # Study period. Keep rows from start_year onward when year is known.
    if "crash_year" in df.columns:
        year = pd.to_numeric(df["crash_year"], errors="coerce")
        df = df[(year.isna()) | (year >= config.start_year)].copy()

    if not vehicles_agg.empty:
        df = df.merge(vehicles_agg, on="collision_id", how="left")
    if not persons_agg.empty:
        df = df.merge(persons_agg, on="collision_id", how="left")

    # Missing aggregates mean the collision appeared in the crash table but not the linked table.
    for c in df.columns:
        if c.startswith("vehicle_") or c.startswith("has_") or c.startswith("driver_") or c.startswith("person_") or c.startswith("safety_") or c.startswith("ped_role_"):
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(0)

    if "crash_month" in df.columns:
        df["season"] = _season_from_month(df["crash_month"])
    else:
        df["season"] = "unknown"

    if "borough" in df.columns:
        df["borough"] = _collapse_rare(df["borough"], top_n=10)
    else:
        df["borough"] = "unknown"

    if "zip_code" in df.columns:
        df["zip_code_group"] = _collapse_rare(df["zip_code"], top_n=60)
    else:
        df["zip_code_group"] = "unknown"

    # Categoricals from crash table. Keep top values only.
    for c in list(df.columns):
        if c.startswith("contributing_factor_vehicle") or c.startswith("vehicle_type_code"):
            df[c] = _collapse_rare(df[c], top_n=config.top_n_categories)

    # Useful person/vehicle ratios.
    if "person_records_count" in df.columns:
        denom = df["person_records_count"].replace(0, np.nan)
        for c in ["person_pedestrian_count", "person_bicyclist_count", "person_occupant_count", "person_child_count", "person_senior_count"]:
            if c in df.columns:
                df[c.replace("_count", "_share")] = df[c] / denom
    if "vehicle_records_count" in df.columns:
        denom_v = df["vehicle_records_count"].replace(0, np.nan)
        for c in ["driver_male_count", "driver_female_count", "driver_unknown_sex_count"]:
            if c in df.columns:
                df[c.replace("_count", "_share")] = df[c] / denom_v

    # Drop rows with invalid target or missing date year if needed.
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype("int8")

    write_parquet_or_pickle(df, config.interim_dir / "03_feature_matrix.parquet")

    leakage_report = pd.DataFrame({
        "excluded_column": sorted([c for c in DIRECT_TARGET_OR_LEAKAGE_COLS.union(RAW_TEXT_EXCLUDE) if c in df.columns]),
        "reason": ["direct target/leakage or raw high-cardinality text"] * len([c for c in DIRECT_TARGET_OR_LEAKAGE_COLS.union(RAW_TEXT_EXCLUDE) if c in df.columns]),
    })
    write_table(leakage_report, config.outputs_dir / "tables" / "03_leakage_exclusion_report.csv", config.outputs_dir / "latex_tables" / "03_leakage_exclusion_report.tex")
    write_text("\n".join(["Leakage exclusion report", "========================", ""] + [f"- {r.excluded_column}: {r.reason}" for r in leakage_report.itertuples()]), config.outputs_dir / "reports" / "03_leakage_exclusion_report.txt")
    return df


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set([ID_COL, TARGET_COL, "crash_date"]).union(DIRECT_TARGET_OR_LEAKAGE_COLS).union(RAW_TEXT_EXCLUDE)
    excluded.update([c for c in df.columns if c.endswith("_sum") and c in DIRECT_TARGET_OR_LEAKAGE_COLS])
    # Keep crash_year for temporal splitting, not modeling.
    excluded.add("crash_year")
    features = [c for c in df.columns if c not in excluded]
    # Avoid all-null features.
    features = [c for c in features if df[c].notna().any()]
    return features


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    all_features = get_model_feature_columns(df)
    vehicle_features = [
        c for c in all_features
        if c.startswith("vehicle_") or c.startswith("driver_") or c.endswith("_vehicle") or c.endswith("_precrash")
    ]
    person_features = [c for c in all_features if c.startswith("person_") or c.startswith("safety_") or c.startswith("ped_role_")]
    crash_features = [c for c in all_features if c not in set(vehicle_features).union(person_features)]
    return {
        "crash_only": crash_features,
        "crash_vehicle": crash_features + vehicle_features,
        "crash_person": crash_features + person_features,
        "full": all_features,
    }


def dataset_summary(config: ProjectConfig, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"metric": "rows", "value": len(df)})
    rows.append({"metric": "columns", "value": df.shape[1]})
    rows.append({"metric": "injury_or_fatality_rate", "value": float(df[TARGET_COL].mean())})
    if "crash_year" in df.columns:
        rows.append({"metric": "min_year", "value": int(pd.to_numeric(df["crash_year"], errors="coerce").min())})
        rows.append({"metric": "max_year", "value": int(pd.to_numeric(df["crash_year"], errors="coerce").max())})
    if "vehicle_records_count" in df.columns:
        rows.append({"metric": "mean_vehicle_records_per_collision", "value": float(df["vehicle_records_count"].mean())})
    if "person_records_count" in df.columns:
        rows.append({"metric": "mean_person_records_per_collision", "value": float(df["person_records_count"].mean())})
    summary = pd.DataFrame(rows)
    write_table(summary, config.outputs_dir / "tables" / "03_dataset_summary.csv", config.outputs_dir / "latex_tables" / "03_dataset_summary.tex")
    return summary
