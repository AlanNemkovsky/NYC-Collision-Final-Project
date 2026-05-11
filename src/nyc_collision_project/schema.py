from __future__ import annotations

import re
from typing import Iterable, Optional


def normalize_col(name: str) -> str:
    """Normalize a raw CSV column name to a predictable snake_case key."""
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalized_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {normalize_col(c): c for c in columns}


def find_col(columns: Iterable[str], aliases: Iterable[str], required: bool = False) -> Optional[str]:
    lookup = normalized_lookup(columns)
    alias_norms = [normalize_col(a) for a in aliases]
    for a in alias_norms:
        if a in lookup:
            return lookup[a]
    if required:
        raise KeyError(f"Missing required column. Tried aliases: {list(aliases)}")
    return None


def find_cols_by_prefix(columns: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    lookup = normalized_lookup(columns)
    out: list[str] = []
    for norm, raw in lookup.items():
        if any(norm.startswith(normalize_col(p)) for p in prefixes):
            out.append(raw)
    return out


ALIASES = {
    "collision_id": ["COLLISION_ID", "collision id", "collisionid"],
    "crash_date": ["CRASH DATE", "CRASH_DATE", "crash date"],
    "crash_time": ["CRASH TIME", "CRASH_TIME", "crash time"],
    "borough": ["BOROUGH"],
    "zip_code": ["ZIP CODE", "ZIP_CODE", "zipcode", "zip"],
    "latitude": ["LATITUDE"],
    "longitude": ["LONGITUDE"],
    "location": ["LOCATION"],
    "on_street_name": ["ON STREET NAME", "ON_STREET_NAME"],
    "cross_street_name": ["CROSS STREET NAME", "CROSS_STREET_NAME"],
    "off_street_name": ["OFF STREET NAME", "OFF_STREET_NAME"],
    "persons_injured": ["NUMBER OF PERSONS INJURED", "NUMBER_OF_PERSONS_INJURED"],
    "persons_killed": ["NUMBER OF PERSONS KILLED", "NUMBER_OF_PERSONS_KILLED"],
    "pedestrians_injured": ["NUMBER OF PEDESTRIANS INJURED", "NUMBER_OF_PEDESTRIANS_INJURED"],
    "pedestrians_killed": ["NUMBER OF PEDESTRIANS KILLED", "NUMBER_OF_PEDESTRIANS_KILLED"],
    "cyclist_injured": ["NUMBER OF CYCLIST INJURED", "NUMBER_OF_CYCLIST_INJURED", "NUMBER OF CYCLISTS INJURED"],
    "cyclist_killed": ["NUMBER OF CYCLIST KILLED", "NUMBER_OF_CYCLIST_KILLED", "NUMBER OF CYCLISTS KILLED"],
    "motorist_injured": ["NUMBER OF MOTORIST INJURED", "NUMBER_OF_MOTORIST_INJURED", "NUMBER OF MOTORISTS INJURED"],
    "motorist_killed": ["NUMBER OF MOTORIST KILLED", "NUMBER_OF_MOTORIST_KILLED", "NUMBER OF MOTORISTS KILLED"],
    "vehicle_id": ["VEHICLE_ID", "vehicle id"],
    "vehicle_type": ["VEHICLE_TYPE", "VEHICLE TYPE", "VEHICLE BODY TYPE"],
    "vehicle_year": ["VEHICLE_YEAR", "VEHICLE YEAR"],
    "vehicle_occupants": ["VEHICLE_OCCUPANTS", "VEHICLE OCCUPANTS"],
    "state_registration": ["STATE_REGISTRATION", "STATE REGISTRATION"],
    "travel_direction": ["TRAVEL_DIRECTION", "TRAVEL DIRECTION"],
    "pre_crash": ["PRE_CRASH", "PRE CRASH"],
    "driver_sex": ["DRIVER_SEX", "DRIVER SEX"],
    "driver_license_status": ["DRIVER_LICENSE_STATUS", "DRIVER LICENSE STATUS"],
    "person_id": ["PERSON_ID", "PERSON ID"],
    "person_type": ["PERSON_TYPE", "PERSON TYPE"],
    "person_injury": ["PERSON_INJURY", "PERSON INJURY"],
    "person_age": ["PERSON_AGE", "PERSON AGE"],
    "person_sex": ["PERSON_SEX", "PERSON SEX"],
    "ejection": ["EJECTION"],
    "emotional_status": ["EMOTIONAL_STATUS", "EMOTIONAL STATUS"],
    "bodily_injury": ["BODILY_INJURY", "BODILY INJURY"],
    "position_in_vehicle": ["POSITION_IN_VEHICLE", "POSITION IN VEHICLE"],
    "safety_equipment": ["SAFETY_EQUIPMENT", "SAFETY EQUIPMENT"],
    "ped_location": ["PED_LOCATION", "PED LOCATION"],
    "ped_action": ["PED_ACTION", "PED ACTION"],
    "complaint": ["COMPLAINT"],
    "ped_role": ["PED_ROLE", "PED ROLE"],
}


def get_alias(columns: Iterable[str], key: str, required: bool = False) -> Optional[str]:
    return find_col(columns, ALIASES[key], required=required)


def standardize_columns(df):
    """Return a copy with normalized column names."""
    out = df.copy()
    out.columns = [normalize_col(c) for c in out.columns]
    return out
