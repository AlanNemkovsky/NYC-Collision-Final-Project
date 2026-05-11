from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DOWNLOADS = Path(r"C:\Users\alann\Downloads")


@dataclass
class ProjectConfig:
    crashes_path: Path = DEFAULT_DOWNLOADS / "Motor_Vehicle_Collisions_-_Crashes.csv"
    vehicles_path: Path = DEFAULT_DOWNLOADS / "Motor_Vehicle_Collisions_-_Vehicles.csv"
    persons_path: Path = DEFAULT_DOWNLOADS / "Motor_Vehicle_Collisions_-_Person.csv"
    outputs_dir: Path = Path("outputs")
    interim_dir: Path = Path("data/interim")
    chunk_size: int = 250_000
    start_year: int = 2021
    preferred_test_year: int = 2025
    max_train_rows: int = 300_000
    max_test_rows: int = 100_000
    random_state: int = 42
    top_n_categories: int = 35
    enable_optional_xgboost: bool = True
    enable_optional_shap: bool = True
    skip_clustering: bool = False
    make_outputs_zip: bool = True

    def create_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["figures", "tables", "latex_tables", "reports", "models"]:
            (self.outputs_dir / sub).mkdir(parents=True, exist_ok=True)
