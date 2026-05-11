from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .io_utils import read_header, safe_read_csv, read_csv_chunks, write_parquet_or_pickle, write_text, write_table
from .schema import get_alias, standardize_columns, normalize_col


def _select_existing(header: list[str], keys: list[str]) -> list[str]:
    cols = []
    for key in keys:
        col = get_alias(header, key, required=False)
        if col and col not in cols:
            cols.append(col)
    return cols


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clean_string(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    )


def _lower_clean(s: pd.Series) -> pd.Series:
    return _clean_string(s).str.lower()


def clean_crashes(config: ProjectConfig) -> pd.DataFrame:
    header = read_header(config.crashes_path)
    base_keys = [
        "collision_id", "crash_date", "crash_time", "borough", "zip_code", "latitude", "longitude",
        "location", "on_street_name", "cross_street_name", "off_street_name",
        "persons_injured", "persons_killed", "pedestrians_injured", "pedestrians_killed",
        "cyclist_injured", "cyclist_killed", "motorist_injured", "motorist_killed",
    ]
    usecols = _select_existing(header, base_keys)
    # Add repeated contributing factor and vehicle type code columns.
    norm_to_raw = {normalize_col(c): c for c in header}
    for norm, raw in norm_to_raw.items():
        if norm.startswith("contributing_factor_vehicle") or norm.startswith("vehicle_type_code"):
            if raw not in usecols:
                usecols.append(raw)

    df = safe_read_csv(config.crashes_path, usecols=usecols)
    df = standardize_columns(df)

    if "collision_id" not in df.columns:
        raise KeyError("The crashes file must contain COLLISION_ID.")

    df["collision_id"] = _numeric(df["collision_id"]).astype("Int64")
    df = df.dropna(subset=["collision_id"]).copy()
    df["collision_id"] = df["collision_id"].astype("int64")

    if "crash_date" in df.columns:
        df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    else:
        df["crash_date"] = pd.NaT

    if "crash_time" in df.columns:
        time_str = _clean_string(df["crash_time"]).fillna("00:00")
        hours = pd.to_numeric(time_str.str.extract(r"^(\d{1,2})", expand=False), errors="coerce")
        minutes = pd.to_numeric(time_str.str.extract(r":(\d{1,2})", expand=False), errors="coerce")
        df["crash_hour"] = hours.clip(0, 23).fillna(0).astype("int16")
        df["crash_minute"] = minutes.clip(0, 59).fillna(0).astype("int16")
    else:
        df["crash_hour"] = 0
        df["crash_minute"] = 0

    for c in ["borough", "zip_code", "location", "on_street_name", "cross_street_name", "off_street_name"]:
        if c in df.columns:
            df[c] = _clean_string(df[c])

    for c in ["latitude", "longitude"]:
        if c in df.columns:
            df[c] = _numeric(df[c])

    injury_cols = [
        "number_of_persons_injured", "number_of_persons_killed", "number_of_pedestrians_injured",
        "number_of_pedestrians_killed", "number_of_cyclist_injured", "number_of_cyclist_killed",
        "number_of_motorist_injured", "number_of_motorist_killed",
        "persons_injured", "persons_killed", "pedestrians_injured", "pedestrians_killed",
        "cyclist_injured", "cyclist_killed", "motorist_injured", "motorist_killed",
    ]
    actual_injury_cols = [c for c in injury_cols if c in df.columns]
    for c in actual_injury_cols:
        df[c] = _numeric(df[c]).fillna(0).clip(lower=0)

    if actual_injury_cols:
        df["total_injury_fatality_count"] = df[actual_injury_cols].sum(axis=1)
        df["injury_or_fatality"] = (df["total_injury_fatality_count"] > 0).astype("int8")
    else:
        raise KeyError("Could not find injury/fatality count columns in the crashes file.")

    if "crash_date" in df.columns:
        df["crash_year"] = df["crash_date"].dt.year.astype("Int64")
        df["crash_month"] = df["crash_date"].dt.month.astype("Int64")
        df["crash_dayofweek"] = df["crash_date"].dt.dayofweek.astype("Int64")
    else:
        df["crash_year"] = pd.NA
        df["crash_month"] = pd.NA
        df["crash_dayofweek"] = pd.NA

    df["is_weekend"] = df["crash_dayofweek"].isin([5, 6]).astype("int8")
    df["is_rush_hour"] = df["crash_hour"].isin([7, 8, 9, 16, 17, 18]).astype("int8")
    df["is_late_night"] = df["crash_hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")
    df["has_coordinates"] = ((df.get("latitude").notna() if "latitude" in df.columns else False) & (df.get("longitude").notna() if "longitude" in df.columns else False)).astype("int8")
    df["has_cross_street"] = df.get("cross_street_name", pd.Series(pd.NA, index=df.index)).notna().astype("int8")
    df["has_off_street"] = df.get("off_street_name", pd.Series(pd.NA, index=df.index)).notna().astype("int8")

    for c in df.columns:
        if c.startswith("contributing_factor_vehicle") or c.startswith("vehicle_type_code"):
            df[c] = _lower_clean(df[c])

    write_parquet_or_pickle(df, config.interim_dir / "02_clean_crashes.parquet")
    return df


def _combine_chunk_aggs(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    all_aggs = pd.concat(parts, axis=0)
    sum_cols = [c for c in all_aggs.columns if not c.startswith("has_") and not c.endswith("_max")]
    max_cols = [c for c in all_aggs.columns if c.startswith("has_") or c.endswith("_max")]
    grouped = []
    if sum_cols:
        grouped.append(all_aggs[sum_cols].groupby(level=0).sum())
    if max_cols:
        grouped.append(all_aggs[max_cols].groupby(level=0).max())
    if not grouped:
        return pd.DataFrame(index=all_aggs.index.unique())
    return pd.concat(grouped, axis=1).reset_index().rename(columns={"index": "collision_id"})


def clean_and_aggregate_vehicles(config: ProjectConfig) -> pd.DataFrame:
    header = read_header(config.vehicles_path)
    keys = [
        "collision_id", "vehicle_type", "vehicle_year", "vehicle_occupants", "state_registration",
        "travel_direction", "pre_crash", "driver_sex", "driver_license_status",
    ]
    usecols = _select_existing(header, keys)
    norm_to_raw = {normalize_col(c): c for c in header}
    for norm, raw in norm_to_raw.items():
        if norm.startswith("contributing_factor") and raw not in usecols:
            usecols.append(raw)

    cid_raw = get_alias(header, "collision_id", required=True)
    parts = []
    for chunk in read_csv_chunks(config.vehicles_path, chunksize=config.chunk_size, usecols=usecols):
        chunk = standardize_columns(chunk)
        if "collision_id" not in chunk.columns:
            continue
        chunk["collision_id"] = _numeric(chunk["collision_id"])
        chunk = chunk.dropna(subset=["collision_id"]).copy()
        chunk["collision_id"] = chunk["collision_id"].astype("int64")
        out = pd.DataFrame(index=chunk.index)
        out["collision_id"] = chunk["collision_id"]
        out["vehicle_records_count"] = 1

        if "vehicle_occupants" in chunk.columns:
            occ = _numeric(chunk["vehicle_occupants"]).clip(lower=0, upper=100)
            out["vehicle_occupants_sum"] = occ.fillna(0)
            out["vehicle_occupants_valid_count"] = occ.notna().astype("int8")
            out["vehicle_occupants_max"] = occ.fillna(0)

        type_cols = [c for c in ["vehicle_type"] if c in chunk.columns]
        if type_cols:
            vt = _lower_clean(chunk[type_cols[0]]).fillna("")
            patterns = {
                "has_sedan_vehicle": r"sedan",
                "has_suv_vehicle": r"sport utility|suv|station wagon",
                "has_truck_vehicle": r"truck|tractor|box|freight|pick-up|pickup|dump",
                "has_bus_vehicle": r"bus",
                "has_taxi_vehicle": r"taxi",
                "has_bike_vehicle": r"bike|bicycle|bicycl|e-bike",
                "has_motorcycle_vehicle": r"motorcycle|moped|scooter",
                "has_emergency_vehicle": r"ambulance|fire|police",
            }
            for name, pat in patterns.items():
                out[name] = vt.str.contains(pat, regex=True, na=False).astype("int8")

        if "pre_crash" in chunk.columns:
            pc = _lower_clean(chunk["pre_crash"]).fillna("")
            for name, pat in {
                "has_parked_precrash": r"parked",
                "has_turning_precrash": r"turn|making",
                "has_merging_precrash": r"merg",
                "has_backing_precrash": r"back",
                "has_stopped_precrash": r"stopped",
                "has_going_straight_precrash": r"going straight",
            }.items():
                out[name] = pc.str.contains(pat, regex=True, na=False).astype("int8")

        if "driver_sex" in chunk.columns:
            ds = _lower_clean(chunk["driver_sex"]).fillna("")
            out["driver_male_count"] = ds.isin(["m", "male"]).astype("int8")
            out["driver_female_count"] = ds.isin(["f", "female"]).astype("int8")
            out["driver_unknown_sex_count"] = (~ds.isin(["m", "male", "f", "female"]) | (ds == "")).astype("int8")

        if "vehicle_year" in chunk.columns:
            vy = _numeric(chunk["vehicle_year"])
            out["vehicle_year_valid_count"] = vy.between(1950, 2030).astype("int8")
            out["vehicle_year_sum"] = vy.where(vy.between(1950, 2030), np.nan).fillna(0)

        agg = out.groupby("collision_id").agg({c: ("max" if c.startswith("has_") or c.endswith("_max") else "sum") for c in out.columns if c != "collision_id"})
        parts.append(agg)

    vehicles_agg = _combine_chunk_aggs(parts)
    if not vehicles_agg.empty:
        if "vehicle_occupants_valid_count" in vehicles_agg.columns:
            vehicles_agg["vehicle_occupants_mean"] = vehicles_agg["vehicle_occupants_sum"] / vehicles_agg["vehicle_occupants_valid_count"].replace(0, np.nan)
        if "vehicle_year_valid_count" in vehicles_agg.columns:
            vehicles_agg["vehicle_year_mean"] = vehicles_agg["vehicle_year_sum"] / vehicles_agg["vehicle_year_valid_count"].replace(0, np.nan)
        for c in vehicles_agg.columns:
            if c != "collision_id" and vehicles_agg[c].dtype == "float64":
                vehicles_agg[c] = vehicles_agg[c].astype("float32")
    write_parquet_or_pickle(vehicles_agg, config.interim_dir / "02_vehicle_collision_agg.parquet")
    return vehicles_agg


def clean_and_aggregate_persons(config: ProjectConfig) -> pd.DataFrame:
    header = read_header(config.persons_path)
    keys = [
        "collision_id", "person_type", "person_age", "person_sex", "position_in_vehicle",
        "safety_equipment", "ped_location", "ped_action", "ped_role",
    ]
    usecols = _select_existing(header, keys)
    parts = []
    for chunk in read_csv_chunks(config.persons_path, chunksize=config.chunk_size, usecols=usecols):
        chunk = standardize_columns(chunk)
        if "collision_id" not in chunk.columns:
            continue
        chunk["collision_id"] = _numeric(chunk["collision_id"])
        chunk = chunk.dropna(subset=["collision_id"]).copy()
        chunk["collision_id"] = chunk["collision_id"].astype("int64")
        out = pd.DataFrame(index=chunk.index)
        out["collision_id"] = chunk["collision_id"]
        out["person_records_count"] = 1

        if "person_type" in chunk.columns:
            pt = _lower_clean(chunk["person_type"]).fillna("")
            out["person_occupant_count"] = pt.str.contains("occupant|registrant|passenger|driver", regex=True, na=False).astype("int8")
            out["person_pedestrian_count"] = pt.str.contains("pedestrian", regex=True, na=False).astype("int8")
            out["person_bicyclist_count"] = pt.str.contains("bicyclist|cyclist|bike", regex=True, na=False).astype("int8")
            out["person_other_type_count"] = (~(out["person_occupant_count"].astype(bool) | out["person_pedestrian_count"].astype(bool) | out["person_bicyclist_count"].astype(bool))).astype("int8")

        if "person_age" in chunk.columns:
            age = _numeric(chunk["person_age"])
            valid = age.between(0, 110)
            out["person_age_sum"] = age.where(valid, np.nan).fillna(0)
            out["person_age_valid_count"] = valid.astype("int8")
            out["person_child_count"] = age.between(0, 17).fillna(False).astype("int8")
            out["person_senior_count"] = age.ge(65).fillna(False).astype("int8")

        if "person_sex" in chunk.columns:
            ps = _lower_clean(chunk["person_sex"]).fillna("")
            out["person_male_count"] = ps.isin(["m", "male"]).astype("int8")
            out["person_female_count"] = ps.isin(["f", "female"]).astype("int8")
            out["person_unknown_sex_count"] = (~ps.isin(["m", "male", "f", "female"]) | (ps == "")).astype("int8")

        if "safety_equipment" in chunk.columns:
            se = _lower_clean(chunk["safety_equipment"]).fillna("")
            out["safety_belt_count"] = se.str.contains("belt|harness", regex=True, na=False).astype("int8")
            out["safety_none_unknown_count"] = se.str.contains("none|unknown|does not", regex=True, na=False).astype("int8")

        if "ped_role" in chunk.columns:
            pr = _lower_clean(chunk["ped_role"]).fillna("")
            out["ped_role_driver_count"] = pr.str.contains("driver", na=False).astype("int8")
            out["ped_role_passenger_count"] = pr.str.contains("passenger", na=False).astype("int8")

        agg = out.groupby("collision_id").agg({c: "sum" for c in out.columns if c != "collision_id"})
        parts.append(agg)

    persons_agg = _combine_chunk_aggs(parts)
    if not persons_agg.empty:
        if "person_age_valid_count" in persons_agg.columns:
            persons_agg["person_age_mean"] = persons_agg["person_age_sum"] / persons_agg["person_age_valid_count"].replace(0, np.nan)
        for c in persons_agg.columns:
            if c != "collision_id" and persons_agg[c].dtype == "float64":
                persons_agg[c] = persons_agg[c].astype("float32")
    write_parquet_or_pickle(persons_agg, config.interim_dir / "02_person_collision_agg.parquet")
    return persons_agg


def clean_all(config: ProjectConfig) -> dict[str, pd.DataFrame]:
    crashes = clean_crashes(config)
    vehicles_agg = clean_and_aggregate_vehicles(config)
    persons_agg = clean_and_aggregate_persons(config)

    reports_dir = config.outputs_dir / "reports"
    lines = [
        "Cleaning summary",
        "================",
        f"Clean crashes rows: {len(crashes):,}",
        f"Vehicle aggregate rows: {len(vehicles_agg):,}",
        f"Person aggregate rows: {len(persons_agg):,}",
        "",
        "Leakage control note:",
        "Direct person-level injury descriptors such as PERSON_INJURY, BODILY_INJURY, COMPLAINT, EJECTION, and EMOTIONAL_STATUS are not used as prediction features.",
        "Crash-table injury/fatality counts are used only to construct the target and are excluded from model features.",
    ]
    write_text("\n".join(lines), reports_dir / "02_cleaning_summary.txt")
    return {"crashes": crashes, "vehicles_agg": vehicles_agg, "persons_agg": persons_agg}
