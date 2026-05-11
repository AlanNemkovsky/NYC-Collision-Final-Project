from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import ProjectConfig
from .features import TARGET_COL, feature_groups
from .io_utils import write_json, write_table, write_text


plt.switch_backend("Agg")


def _make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)



def _sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable missing values to numpy NaN for scikit-learn compatibility."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype.name in ("string", "category"):
            X[col] = X[col].astype(object)
    return X.where(pd.notna(X), np.nan)

def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in X.columns if X[c].dtype.name in ("object", "string", "category")]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def _linear_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = _split_columns(X)
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _make_one_hot())]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def _tree_onehot_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = _split_columns(X)
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _make_one_hot())]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def _ordinal_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = _split_columns(X)
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def _safe_proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1]
        return p.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    pred = model.predict(X)
    return pred.astype(float)


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": float(np.mean(pred == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate_predicted": float(pred.mean()),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def _sample_by_class(X: pd.DataFrame, y: pd.Series, max_rows: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y
    frac = max_rows / len(X)
    sampled_idx = (
        pd.DataFrame({"y": y}, index=X.index)
        .groupby("y", group_keys=False)
        .sample(frac=frac, random_state=random_state)
        .index
    )
    return X.loc[sampled_idx], y.loc[sampled_idx]


def _temporal_or_fallback_split(df: pd.DataFrame, features: list[str], config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    data = df.dropna(subset=[TARGET_COL]).copy()
    y = data[TARGET_COL].astype(int)
    X = _sanitize_features(data[features])
    split_note = ""

    if "crash_year" in data.columns:
        years = pd.to_numeric(data["crash_year"], errors="coerce")
        valid_years = sorted([int(y) for y in years.dropna().unique()])
        if config.preferred_test_year in valid_years and (years < config.preferred_test_year).sum() > 100:
            train_idx = data.index[years < config.preferred_test_year]
            test_idx = data.index[years == config.preferred_test_year]
            split_note = f"Temporal split: train years before {config.preferred_test_year}; test year {config.preferred_test_year}."
            return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx], split_note
        elif len(valid_years) >= 2:
            test_year = max(valid_years)
            train_idx = data.index[years < test_year]
            test_idx = data.index[years == test_year]
            split_note = f"Temporal fallback split: train years before {test_year}; test year {test_year}."
            return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx], split_note

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y if y.nunique() == 2 else None, random_state=config.random_state)
    split_note = "Random stratified fallback split because no usable multi-year crash_year field was available."
    return X_train, X_test, y_train, y_test, split_note


def _build_models(X: pd.DataFrame, config: ProjectConfig) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "dummy_most_frequent": Pipeline([
            ("preprocess", _linear_preprocessor(X)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_l2_balanced": Pipeline([
            ("preprocess", _linear_preprocessor(X)),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=config.random_state)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", _tree_onehot_preprocessor(X)),
            ("model", RandomForestClassifier(n_estimators=80, max_depth=16, min_samples_leaf=10, class_weight="balanced_subsample", n_jobs=2, random_state=config.random_state)),
        ]),
        "extra_trees": Pipeline([
            ("preprocess", _tree_onehot_preprocessor(X)),
            ("model", ExtraTreesClassifier(n_estimators=100, max_depth=18, min_samples_leaf=8, class_weight="balanced", n_jobs=2, random_state=config.random_state)),
        ]),
    }
    if config.enable_optional_xgboost:
        try:
            from xgboost import XGBClassifier  # type: ignore
            models["xgboost_optional"] = Pipeline([
                ("preprocess", _tree_onehot_preprocessor(X)),
                ("model", XGBClassifier(
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=2,
                    random_state=config.random_state,
                )),
            ])
        except Exception as e:
            warnings.warn(f"XGBoost not available; skipping optional model. Reason: {e}")
    return models


def _plot_curves(config: ProjectConfig, curve_data: dict[str, dict[str, np.ndarray]]):
    plt.figure(figsize=(6.2, 5))
    for name, d in curve_data.items():
        if len(np.unique(d["y"])) < 2:
            continue
        fpr, tpr, _ = roc_curve(d["y"], d["scores"])
        auc = roc_auc_score(d["y"], d["scores"])
        plt.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on held-out test set")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(config.outputs_dir / "figures" / "fig_roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.2, 5))
    for name, d in curve_data.items():
        if len(np.unique(d["y"])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(d["y"], d["scores"])
        ap = average_precision_score(d["y"], d["scores"])
        plt.plot(recall, precision, label=f"{name} ({ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves on held-out test set")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(config.outputs_dir / "figures" / "fig_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close()


def _plot_confusion(config: ProjectConfig, y_true: np.ndarray, scores: np.ndarray, threshold: float, model_name: str):
    cm = confusion_matrix(y_true, (scores >= threshold).astype(int))
    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(cm)
    plt.title(f"Confusion matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.xticks([0, 1], ["No injury", "Injury/fatality"], rotation=15)
    plt.yticks([0, 1], ["No injury", "Injury/fatality"])
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(config.outputs_dir / "figures" / "fig_confusion_matrix_best.png", dpi=180, bbox_inches="tight")
    plt.close()


def _feature_names_from_pipeline(pipe: Pipeline, input_cols: list[str]) -> list[str]:
    pre = pipe.named_steps.get("preprocess")
    if hasattr(pre, "get_feature_names_out"):
        try:
            return list(pre.get_feature_names_out())
        except Exception:
            pass
    return input_cols


def _plot_logreg_coefficients(config: ProjectConfig, pipe: Pipeline, input_cols: list[str]) -> pd.DataFrame | None:
    if "model" not in pipe.named_steps or not hasattr(pipe.named_steps["model"], "coef_"):
        return None
    names = _feature_names_from_pipeline(pipe, input_cols)
    coefs = pipe.named_steps["model"].coef_.ravel()
    n = min(len(names), len(coefs))
    table = pd.DataFrame({"feature": names[:n], "coefficient": coefs[:n]})
    table["abs_coefficient"] = table["coefficient"].abs()
    top = table.sort_values("abs_coefficient", ascending=False).head(25).sort_values("coefficient")
    plt.figure(figsize=(8, 7))
    plt.barh(top["feature"], top["coefficient"])
    plt.title("Largest regularized logistic regression coefficients")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(config.outputs_dir / "figures" / "fig_logreg_coefficients.png", dpi=180, bbox_inches="tight")
    plt.close()
    write_table(table.sort_values("abs_coefficient", ascending=False), config.outputs_dir / "tables" / "logreg_coefficients.csv", config.outputs_dir / "latex_tables" / "logreg_coefficients.tex")
    return table


def _plot_permutation_importance(config: ProjectConfig, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> pd.DataFrame | None:
    if len(X_test) == 0 or y_test.nunique() < 2:
        return None
    n = min(1000, len(X_test))
    Xs = X_test.sample(n=n, random_state=config.random_state) if len(X_test) > n else X_test
    ys = y_test.loc[Xs.index]
    try:
        result = permutation_importance(pipe, Xs, ys, scoring="average_precision", n_repeats=1, random_state=config.random_state, n_jobs=1)
    except Exception as e:
        warnings.warn(f"Could not compute permutation importance: {e}")
        return None
    imp = pd.DataFrame({
        "feature": list(X_test.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    top = imp.head(20).sort_values("importance_mean")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.title(f"Permutation importance: {model_name}")
    plt.xlabel("Mean decrease in average precision")
    plt.tight_layout()
    plt.savefig(config.outputs_dir / "figures" / "fig_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close()
    write_table(imp, config.outputs_dir / "tables" / "feature_importance.csv", config.outputs_dir / "latex_tables" / "feature_importance.tex")
    return imp


def train_and_evaluate(config: ProjectConfig, df: pd.DataFrame) -> dict[str, Any]:
    groups = feature_groups(df)
    full_features = groups["full"]
    X_train, X_test, y_train, y_test, split_note = _temporal_or_fallback_split(df, full_features, config)
    X_train, y_train = _sample_by_class(X_train, y_train, config.max_train_rows, config.random_state)
    X_test, y_test = _sample_by_class(X_test, y_test, config.max_test_rows, config.random_state)

    # Train full-set model comparison.
    models = _build_models(X_train, config)
    rows = []
    curve_data: dict[str, dict[str, np.ndarray]] = {}
    trained: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train if y_train.nunique() == 2 else None, random_state=config.random_state
    ) if len(X_train) > 1000 and y_train.nunique() == 2 else (X_train, X_train, y_train, y_train)

    for name, pipe in models.items():
        try:
            pipe.fit(X_fit, y_fit)
            val_scores = _safe_proba(pipe, X_val)
            threshold = _best_threshold(y_val.to_numpy(), val_scores)
            # Refit on all sampled training data after threshold selection.
            pipe.fit(X_train, y_train)
            test_scores = _safe_proba(pipe, X_test)
            row = {"model": name, "feature_set": "full", "train_rows": len(X_train), "test_rows": len(X_test)}
            row.update(_metrics(y_test.to_numpy(), test_scores, threshold))
            rows.append(row)
            curve_data[name] = {"y": y_test.to_numpy(), "scores": test_scores}
            trained[name] = pipe
            thresholds[name] = threshold
            joblib.dump(pipe, config.outputs_dir / "models" / f"{name}.joblib")
        except Exception as e:
            warnings.warn(f"Model {name} failed and was skipped: {e}")
            rows.append({"model": name, "feature_set": "full", "error": str(e), "train_rows": len(X_train), "test_rows": len(X_test)})

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.sort_values(["pr_auc", "roc_auc"], ascending=False, na_position="last") if "pr_auc" in metrics_df.columns else metrics_df
    write_table(metrics_df, config.outputs_dir / "tables" / "model_metrics.csv", config.outputs_dir / "latex_tables" / "model_metrics.tex")

    # Ablation uses a regularized logistic baseline for efficiency and interpretability.
    ablation_rows = []
    for group_name, feats in groups.items():
        if not feats:
            continue
        Xg_train, Xg_test, yg_train, yg_test, _ = _temporal_or_fallback_split(df, feats, config)
        Xg_train, yg_train = _sample_by_class(Xg_train, yg_train, config.max_train_rows, config.random_state)
        Xg_test, yg_test = _sample_by_class(Xg_test, yg_test, config.max_test_rows, config.random_state)
        pipe = Pipeline([
            ("preprocess", _linear_preprocessor(Xg_train)),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=config.random_state)),
        ])
        try:
            pipe.fit(Xg_train, yg_train)
            scores = _safe_proba(pipe, Xg_test)
            threshold = _best_threshold(yg_test.to_numpy(), scores)
            row = {"feature_set": group_name, "model": "logistic_l2_balanced", "n_features_raw": len(feats), "train_rows": len(Xg_train), "test_rows": len(Xg_test)}
            row.update(_metrics(yg_test.to_numpy(), scores, threshold))
            ablation_rows.append(row)
        except Exception as e:
            ablation_rows.append({"feature_set": group_name, "model": "logistic_l2_balanced", "error": str(e), "n_features_raw": len(feats)})
    ablation_df = pd.DataFrame(ablation_rows)
    write_table(ablation_df, config.outputs_dir / "tables" / "ablation_metrics.csv", config.outputs_dir / "latex_tables" / "ablation_metrics.tex")

    # Plots and model reports.
    if curve_data:
        _plot_curves(config, curve_data)
    best_model_name = None
    valid_metrics = metrics_df.dropna(subset=["pr_auc"]) if "pr_auc" in metrics_df.columns else pd.DataFrame()
    if not valid_metrics.empty:
        best_model_name = valid_metrics.iloc[0]["model"]
        best_pipe = trained.get(best_model_name)
        if best_pipe is not None:
            scores = curve_data[best_model_name]["scores"]
            threshold = thresholds[best_model_name]
            _plot_confusion(config, y_test.to_numpy(), scores, threshold, best_model_name)
            _plot_permutation_importance(config, best_pipe, X_test, y_test, best_model_name)
    if "logistic_l2_balanced" in trained:
        _plot_logreg_coefficients(config, trained["logistic_l2_balanced"], list(X_train.columns))

    manifest = {
        "split_note": split_note,
        "feature_columns_full": full_features,
        "feature_group_sizes": {k: len(v) for k, v in groups.items()},
        "trained_models": list(trained.keys()),
        "best_model_by_pr_auc": best_model_name,
        "train_rows_after_sampling": len(X_train),
        "test_rows_after_sampling": len(X_test),
        "positive_rate_train": float(y_train.mean()) if len(y_train) else None,
        "positive_rate_test": float(y_test.mean()) if len(y_test) else None,
    }
    write_json(manifest, config.outputs_dir / "reports" / "modeling_manifest.json")

    summary_lines = [
        "Modeling summary",
        "================",
        split_note,
        f"Training rows after optional sampling: {len(X_train):,}",
        f"Test rows after optional sampling: {len(X_test):,}",
        f"Positive rate in training set: {float(y_train.mean()):.4f}",
        f"Positive rate in test set: {float(y_test.mean()):.4f}",
        f"Best model by PR-AUC: {best_model_name}",
        "",
        "Model metrics table saved to outputs/tables/model_metrics.csv.",
        "Ablation metrics table saved to outputs/tables/ablation_metrics.csv.",
    ]
    write_text("\n".join(summary_lines), config.outputs_dir / "reports" / "best_model_summary.txt")

    return {
        "metrics": metrics_df.to_dict(orient="records"),
        "ablation": ablation_df.to_dict(orient="records"),
        "manifest": manifest,
    }
