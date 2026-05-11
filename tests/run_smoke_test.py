from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.check_call([sys.executable, str(root / "tests" / "create_tiny_sample_data.py")], cwd=root)
subprocess.check_call([
    sys.executable, str(root / "run_all.py"),
    "--crashes", str(root / "tests" / "sample_data" / "Motor_Vehicle_Collisions_-_Crashes.csv"),
    "--vehicles", str(root / "tests" / "sample_data" / "Motor_Vehicle_Collisions_-_Vehicles.csv"),
    "--persons", str(root / "tests" / "sample_data" / "Motor_Vehicle_Collisions_-_Person.csv"),
    "--outputs", str(root / "outputs_smoke"),
    "--interim", str(root / "data" / "interim_smoke"),
    "--max-train-rows", "5000",
    "--max-test-rows", "5000",
    "--no-xgboost",
    "--no-shap",
], cwd=root)
print("Smoke test completed.")
