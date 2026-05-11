from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "sample_data"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 2500
collision_ids = np.arange(100000, 100000 + n)
years = rng.choice([2021, 2022, 2023, 2024, 2025], size=n, p=[.2,.2,.2,.2,.2])
months = rng.integers(1, 13, size=n)
days = rng.integers(1, 28, size=n)
hours = rng.integers(0, 24, size=n)
inj = rng.binomial(1, 0.22 + 0.08*np.isin(hours, [0,1,2,3,17,18]) + 0.05*(rng.random(n)>.8))

crashes = pd.DataFrame({
    "CRASH DATE": [f"{m}/{d}/{y}" for m,d,y in zip(months, days, years)],
    "CRASH TIME": [f"{h}:{rng.integers(0,60):02d}" for h in hours],
    "BOROUGH": rng.choice(["BROOKLYN", "QUEENS", "MANHATTAN", "BRONX", "STATEN ISLAND", None], size=n),
    "ZIP CODE": rng.choice([11201, 11230, 10001, 10457, 11372, None], size=n),
    "LATITUDE": rng.normal(40.72, .05, size=n),
    "LONGITUDE": rng.normal(-73.95, .05, size=n),
    "ON STREET NAME": rng.choice(["BROADWAY", "QUEENS BLVD", "5 AVE", "ATLANTIC AVE"], size=n),
    "CROSS STREET NAME": rng.choice(["1 ST", "2 ST", "3 ST", None], size=n),
    "OFF STREET NAME": rng.choice([None, "PARKING LOT"], size=n),
    "NUMBER OF PERSONS INJURED": inj,
    "NUMBER OF PERSONS KILLED": np.zeros(n, dtype=int),
    "NUMBER OF PEDESTRIANS INJURED": rng.binomial(1, 0.04, size=n) * inj,
    "NUMBER OF PEDESTRIANS KILLED": np.zeros(n, dtype=int),
    "NUMBER OF CYCLIST INJURED": rng.binomial(1, 0.03, size=n) * inj,
    "NUMBER OF CYCLIST KILLED": np.zeros(n, dtype=int),
    "NUMBER OF MOTORIST INJURED": rng.binomial(1, 0.12, size=n) * inj,
    "NUMBER OF MOTORIST KILLED": np.zeros(n, dtype=int),
    "CONTRIBUTING FACTOR VEHICLE 1": rng.choice(["Driver Inattention/Distraction", "Unspecified", "Following Too Closely", "Unsafe Speed", "Failure to Yield Right-of-Way"], size=n),
    "CONTRIBUTING FACTOR VEHICLE 2": rng.choice(["Unspecified", "Driver Inattention/Distraction", None], size=n),
    "COLLISION_ID": collision_ids,
    "VEHICLE TYPE CODE 1": rng.choice(["Sedan", "Station Wagon/Sport Utility Vehicle", "Taxi", "Bike", "Bus"], size=n),
    "VEHICLE TYPE CODE 2": rng.choice(["Sedan", "Station Wagon/Sport Utility Vehicle", "Box Truck", None], size=n),
})
crashes.to_csv(OUT / "Motor_Vehicle_Collisions_-_Crashes.csv", index=False)

veh_rows = []
for cid in collision_ids:
    k = rng.integers(1, 4)
    for j in range(k):
        veh_rows.append({
            "UNIQUE_ID": f"v{cid}_{j}",
            "COLLISION_ID": cid,
            "CRASH_DATE": "1/1/2025",
            "CRASH_TIME": "12:00",
            "VEHICLE_ID": j,
            "STATE_REGISTRATION": rng.choice(["NY", "NJ", "PA", None]),
            "VEHICLE_TYPE": rng.choice(["Sedan", "Station Wagon/Sport Utility Vehicle", "Taxi", "Bike", "Bus", "Box Truck"]),
            "VEHICLE_YEAR": rng.choice([2005, 2010, 2018, 2022, None]),
            "TRAVEL_DIRECTION": rng.choice(["North", "South", "East", "West", None]),
            "VEHICLE_OCCUPANTS": rng.integers(0, 5),
            "DRIVER_SEX": rng.choice(["M", "F", "U", None]),
            "PRE_CRASH": rng.choice(["Going Straight Ahead", "Parked", "Making Right Turn", "Merging", "Backing"]),
            "CONTRIBUTING_FACTOR_1": rng.choice(["Unspecified", "Driver Inattention/Distraction", "Unsafe Speed"]),
        })
vehicles = pd.DataFrame(veh_rows)
vehicles.to_csv(OUT / "Motor_Vehicle_Collisions_-_Vehicles.csv", index=False)

per_rows = []
for cid in collision_ids:
    k = rng.integers(1, 5)
    for j in range(k):
        per_rows.append({
            "UNIQUE_ID": f"p{cid}_{j}",
            "COLLISION_ID": cid,
            "CRASH_DATE": "1/1/2025",
            "CRASH_TIME": "12:00",
            "PERSON_ID": j,
            "PERSON_TYPE": rng.choice(["Occupant", "Pedestrian", "Bicyclist", "Registrant"]),
            "PERSON_INJURY": rng.choice(["Unspecified", "Injured", "Killed"], p=[.8,.19,.01]),
            "VEHICLE_ID": rng.integers(0, 3),
            "PERSON_AGE": rng.choice(list(range(1, 90)) + [None]),
            "EJECTION": rng.choice(["Not Ejected", "Ejected", None]),
            "EMOTIONAL_STATUS": rng.choice(["Does Not Apply", "Conscious", None]),
            "BODILY_INJURY": rng.choice(["Does Not Apply", "Head", "Back", None]),
            "POSITION_IN_VEHICLE": rng.choice(["Driver", "Front passenger", "Pedestrian", None]),
            "SAFETY_EQUIPMENT": rng.choice(["Lap Belt & Harness", "None", "Unknown", None]),
            "PED_ROLE": rng.choice(["Driver", "Passenger", "Pedestrian", None]),
            "PERSON_SEX": rng.choice(["M", "F", "U", None]),
        })
persons = pd.DataFrame(per_rows)
persons.to_csv(OUT / "Motor_Vehicle_Collisions_-_Person.csv", index=False)

print(f"Created sample data in {OUT}")
