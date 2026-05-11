# NYC Collision Injury Prediction Final Project

This repository contains the completed local research pipeline, generated outputs, final paper, and recording presentation for Alan Nemkovsky's Introduction to Data Science final project.

**Project title:** Predicting Injury-Producing NYC Motor Vehicle Collisions Using Multi-Table Crash, Vehicle, and Person Data  
**Author:** Alan Nemkovsky  
**Section:** 05  
**NetID:** an1078  
**Code:** https://github.com/AlanNemkovsky/NYC-Collision-Final-Project  
**Video:** https://drive.google.com/file/d/1Ck1Ilf0Wi1tFBWO5ak2oMGRYA9pWlKzK/view?usp=sharing

## Final results summary

The project predicts whether an NYC motor vehicle collision produced at least one injury or fatality using linked crash, vehicle, and person records from NYC Open Data.

Main held-out 2025 result:

- Best model: XGBoost
- ROC-AUC: 0.893
- PR-AUC: 0.860
- F1: 0.787
- Precision: 0.742
- Recall: 0.839

Ablation result:

- Crash-only PR-AUC: 0.656
- Crash + vehicle PR-AUC: 0.795
- Crash + person PR-AUC: 0.830
- Full multi-table PR-AUC: 0.846

## Repository contents

```text
run_all.py                         Full local pipeline entry point
requirements.txt                   Required Python packages
src/nyc_collision_project/         Source code
outputs/                           Generated paper-ready figures, tables, and reports
paper/main.tex                     Final NeurIPS-style LaTeX paper
paper/main.pdf                     Final compiled paper
paper/neurips_2026.sty             NeurIPS 2026-compatible style file for this course project
presentation/nyc_collision_final_presentation.pptx
presentation/speaker_notes.md      Recording guide
```

## Raw data setup

The raw CSV files are intentionally **not** included because they are large public data files. The code expects the three files to be downloaded locally at:

```text
C:\Users\alann\Downloads\Motor_Vehicle_Collisions_-_Crashes.csv
C:\Users\alann\Downloads\Motor_Vehicle_Collisions_-_Vehicles.csv
C:\Users\alann\Downloads\Motor_Vehicle_Collisions_-_Person.csv
```

These are the only raw data files required.

## Setup on Windows

Open PowerShell or Command Prompt in this project folder.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Optional stronger model packages:

```bash
python -m pip install xgboost shap
```

The project still runs without XGBoost and SHAP. If they are installed, the pipeline tries to use them. If not, it falls back to scikit-learn models and permutation importance.

## Run the full pipeline

```bash
python run_all.py --max-train-rows 0 --max-test-rows 0
```

For a quicker local test run:

```bash
python run_all.py --max-train-rows 50000 --max-test-rows 25000 --chunk-size 100000
```

For a tiny debug run with synthetic sample data:

```bash
python tests/create_tiny_sample_data.py
python run_all.py --crashes tests/sample_data/Motor_Vehicle_Collisions_-_Crashes.csv --vehicles tests/sample_data/Motor_Vehicle_Collisions_-_Vehicles.csv --persons tests/sample_data/Motor_Vehicle_Collisions_-_Person.csv --outputs outputs_smoke --interim data/interim_smoke --max-train-rows 5000 --max-test-rows 5000
```

## Final paper and presentation

The final paper is located at:

```text
paper/main.pdf
```

The presentation for the recorded video is located at:

```text
presentation/nyc_collision_final_presentation.pptx
```

The final PDF has been recompiled from `paper/main.tex`. The abstract contains the final public GitHub repository link and the Google Drive video recording link. A root-level copy is also included as `final_paper_nyc_collision.pdf`.

## Important GitHub notes

- Do not upload the raw large CSV files to GitHub.
- Do not upload course lecture slides or example papers to GitHub.
- It is okay to upload the code, README, requirements, paper folder, presentation, and generated result summaries/figures.
- The outputs folder excludes large model binaries to keep the repository lightweight.
