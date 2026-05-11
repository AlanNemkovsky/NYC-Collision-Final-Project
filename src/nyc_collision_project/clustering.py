from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .config import ProjectConfig
from .features import TARGET_COL
from .io_utils import write_table, write_text


plt.switch_backend("Agg")


def _simple_kmeans(X: np.ndarray, k: int, random_state: int, n_init: int = 5, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray, float]:
    """Small dependency-light K-Means implementation for group-level clustering."""
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_centers = None
    best_inertia = np.inf
    n = X.shape[0]
    for _ in range(n_init):
        init_idx = rng.choice(n, size=k, replace=False)
        centers = X[init_idx].copy()
        labels = np.zeros(n, dtype=int)
        for _iter in range(max_iter):
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _iter > 0:
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = X[mask].mean(axis=0)
                else:
                    centers[j] = X[rng.integers(0, n)]
        distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia = float(distances[np.arange(n), labels].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers, best_inertia


def run_clustering(config: ProjectConfig, df: pd.DataFrame) -> dict:
    if config.skip_clustering:
        return {"skipped": True, "reason": "skip_clustering flag was set"}

    group_col = "zip_code_group" if "zip_code_group" in df.columns and df["zip_code_group"].nunique(dropna=True) >= 20 else "borough"
    if group_col not in df.columns or df[group_col].nunique(dropna=True) < 3:
        reason = "Not enough geographic groups for clustering."
        write_text(reason, config.outputs_dir / "reports" / "cluster_report.txt")
        return {"skipped": True, "reason": reason}

    work = df.copy()
    work[group_col] = work[group_col].fillna("unknown").astype(str)
    group = work.groupby(group_col)
    agg = pd.DataFrame({
        "collision_count": group.size(),
        "injury_rate": group[TARGET_COL].mean(),
    })
    for c in [
        "is_weekend", "is_rush_hour", "is_late_night", "vehicle_records_count", "person_records_count",
        "person_pedestrian_count", "person_bicyclist_count", "has_truck_vehicle", "has_bus_vehicle", "has_taxi_vehicle",
        "has_bike_vehicle", "has_motorcycle_vehicle",
    ]:
        if c in work.columns:
            agg[f"mean_{c}"] = group[c].mean()

    # Remove tiny groups whose rates are unstable.
    agg = agg[agg["collision_count"] >= max(30, int(0.001 * len(work)))].copy()
    if len(agg) < 3:
        reason = "Not enough groups remained after minimum count filtering."
        write_text(reason, config.outputs_dir / "reports" / "cluster_report.txt")
        return {"skipped": True, "reason": reason}

    X = agg.drop(columns=["collision_count"]).copy()
    X_mat = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_mat)

    max_k = min(8, len(agg) - 1)
    candidate_rows = []
    for k in range(2, max_k + 1):
        try:
            labels, centers, inertia = _simple_kmeans(X_scaled, k=k, random_state=config.random_state + k)
            sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
            candidate_rows.append({"k": k, "inertia": float(inertia), "silhouette": float(sil)})
        except Exception as e:
            warnings.warn(f"KMeans k={k} failed: {e}")
    candidates = pd.DataFrame(candidate_rows)
    write_table(candidates, config.outputs_dir / "tables" / "cluster_k_selection.csv", config.outputs_dir / "latex_tables" / "cluster_k_selection.tex")

    if candidates.empty:
        reason = "KMeans candidate runs failed."
        write_text(reason, config.outputs_dir / "reports" / "cluster_report.txt")
        return {"skipped": True, "reason": reason}

    best_k = int(candidates.sort_values("silhouette", ascending=False).iloc[0]["k"])
    labels, centers, inertia = _simple_kmeans(X_scaled, k=best_k, random_state=config.random_state + best_k)
    agg["cluster"] = labels

    pca = PCA(n_components=2, random_state=config.random_state)
    coords = pca.fit_transform(X_scaled)
    agg["pca1"] = coords[:, 0]
    agg["pca2"] = coords[:, 1]

    profiles = agg.groupby("cluster").mean(numeric_only=True).reset_index()
    profiles["n_groups"] = agg.groupby("cluster").size().values
    write_table(agg.reset_index(), config.outputs_dir / "tables" / "cluster_assignments.csv", config.outputs_dir / "latex_tables" / "cluster_assignments.tex")
    write_table(profiles, config.outputs_dir / "tables" / "cluster_profiles.csv", config.outputs_dir / "latex_tables" / "cluster_profiles.tex")

    plt.figure(figsize=(6.5, 5))
    scatter = plt.scatter(agg["pca1"], agg["pca2"], c=agg["cluster"], s=np.clip(agg["collision_count"] / agg["collision_count"].max() * 250, 25, 250))
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"PCA visualization of {group_col}-level crash clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(config.outputs_dir / "figures" / "fig_pca_clusters.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(candidates["k"], candidates["inertia"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("K-Means elbow curve")
    plt.tight_layout()
    plt.savefig(config.outputs_dir / "figures" / "fig_kmeans_elbow.png", dpi=180, bbox_inches="tight")
    plt.close()

    lines = [
        "Clustering report",
        "=================",
        f"Grouping variable: {group_col}",
        f"Groups clustered: {len(agg):,}",
        f"Selected k by silhouette score: {best_k}",
        f"Explained variance by PCA components: {pca.explained_variance_ratio_[0]:.4f}, {pca.explained_variance_ratio_[1]:.4f}",
        "",
        "Cluster profiles are saved to outputs/tables/cluster_profiles.csv.",
    ]
    write_text("\n".join(lines), config.outputs_dir / "reports" / "cluster_report.txt")
    return {"skipped": False, "group_col": group_col, "best_k": best_k, "n_groups": len(agg)}
