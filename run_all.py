from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make src importable when running without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nyc_collision_project.audit import run_raw_audit
from nyc_collision_project.cleaning import clean_all
from nyc_collision_project.clustering import run_clustering
from nyc_collision_project.config import ProjectConfig
from nyc_collision_project.eda import run_eda
from nyc_collision_project.features import build_feature_matrix, dataset_summary
from nyc_collision_project.modeling import train_and_evaluate
from nyc_collision_project.reports import collect_run_manifest, final_summary, zip_outputs_for_paper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NYC collision final project pipeline.")
    parser.add_argument("--crashes", type=Path, default=ProjectConfig.crashes_path, help="Path to Motor_Vehicle_Collisions_-_Crashes.csv")
    parser.add_argument("--vehicles", type=Path, default=ProjectConfig.vehicles_path, help="Path to Motor_Vehicle_Collisions_-_Vehicles.csv")
    parser.add_argument("--persons", type=Path, default=ProjectConfig.persons_path, help="Path to Motor_Vehicle_Collisions_-_Person.csv")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"), help="Output folder")
    parser.add_argument("--interim", type=Path, default=Path("data/interim"), help="Interim data folder")
    parser.add_argument("--chunk-size", type=int, default=250_000, help="Rows per chunk for large linked tables")
    parser.add_argument("--start-year", type=int, default=2021, help="Keep crash records from this year onward")
    parser.add_argument("--preferred-test-year", type=int, default=2025, help="Preferred held-out temporal test year")
    parser.add_argument("--max-train-rows", type=int, default=300_000, help="Max training rows after temporal split; set 0 for no cap")
    parser.add_argument("--max-test-rows", type=int, default=100_000, help="Max testing rows after temporal split; set 0 for no cap")
    parser.add_argument("--top-n-categories", type=int, default=35, help="Top categories to keep per high-cardinality categorical feature")
    parser.add_argument("--no-xgboost", action="store_true", help="Disable optional XGBoost even if installed")
    parser.add_argument("--no-shap", action="store_true", help="Disable optional SHAP even if installed")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip PCA/K-Means clustering extension")
    parser.add_argument("--no-zip", action="store_true", help="Do not create outputs_for_paper.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(
        crashes_path=args.crashes,
        vehicles_path=args.vehicles,
        persons_path=args.persons,
        outputs_dir=args.outputs,
        interim_dir=args.interim,
        chunk_size=args.chunk_size,
        start_year=args.start_year,
        preferred_test_year=args.preferred_test_year,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        top_n_categories=args.top_n_categories,
        enable_optional_xgboost=not args.no_xgboost,
        enable_optional_shap=not args.no_shap,
        skip_clustering=args.skip_clustering,
        make_outputs_zip=not args.no_zip,
    )
    config.create_dirs()
    started = time.time()
    completed: list[str] = []

    print("Step 1/7: auditing raw CSV files...")
    audit = run_raw_audit(config)
    completed.append("raw_audit")

    print("Step 2/7: cleaning and aggregating the three tables...")
    cleaned = clean_all(config)
    completed.append("cleaning_and_aggregation")

    print("Step 3/7: building leakage-controlled collision-level feature matrix...")
    df = build_feature_matrix(config, cleaned["crashes"], cleaned["vehicles_agg"], cleaned["persons_agg"])
    dataset_summary(config, df)
    completed.append("feature_matrix")

    print("Step 4/7: generating EDA figures and tables...")
    run_eda(config, df)
    completed.append("eda")

    print("Step 5/7: training/evaluating models and ablation experiments...")
    modeling = train_and_evaluate(config, df)
    completed.append("modeling")

    print("Step 6/7: running PCA/K-Means clustering extension...")
    clustering = run_clustering(config, df)
    completed.append("clustering")

    print("Step 7/7: writing final summaries and output zip...")
    manifest = collect_run_manifest(config, started, completed, extra={
        "audit_summary": audit,
        "modeling_summary": modeling.get("manifest", {}),
        "clustering_summary": clustering,
    })
    final_summary(config)
    zip_path = zip_outputs_for_paper(config)
    completed.append("final_reports")

    print("\nDONE")
    print(f"Outputs folder: {config.outputs_dir.resolve()}")
    if zip_path is not None:
        print(f"Output zip created: {zip_path.resolve()}")
    print("Use outputs_for_paper.zip to inspect or archive generated figures, tables, and reports.")


if __name__ == "__main__":
    main()
