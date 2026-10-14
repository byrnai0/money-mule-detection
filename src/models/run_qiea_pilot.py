
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qiea_feature_selector import (
    QIEAConfig,
    QIEAFeatureSelector,
)


ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = ROOT / "data" / "graphs" / "training_graph.pt"
MANIFEST_FILE = ROOT / "data" / "processed" / "feature_manifest.csv"

OUTPUT_DIR = (
    ROOT / "data" / "graphs" / "evaluation" / "phase9_qiea"
)

SEED = 42
TRAIN_SAMPLE_SIZE = 100_000

# Keep this first pilot small. Increase only after inspecting results.
CONFIG = QIEAConfig(
    population_size=10,
    generations=8,
    min_features=5,
    max_features=22,
    rotation_step=0.03 * np.pi,
    random_seed=SEED,
)


def stratified_indices(indices, labels, size, seed):
    """Sample node indices while preserving class proportions."""
    rng = np.random.default_rng(seed)

    if size >= len(indices):
        return indices.copy()

    local_y = labels[indices]
    selected = []

    for cls in np.unique(local_y):
        class_indices = indices[local_y == cls]
        count = int(round(size * len(class_indices) / len(indices)))
        count = min(count, len(class_indices))

        selected.extend(
            rng.choice(class_indices, size=count, replace=False).tolist()
        )

    selected = np.asarray(selected, dtype=np.int64)

    # Correct rounding so the requested size is respected.
    if len(selected) > size:
        selected = rng.choice(selected, size=size, replace=False)
    elif len(selected) < size:
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        extra_count = min(size - len(selected), len(remaining))
        extra = rng.choice(remaining, size=extra_count, replace=False)
        selected = np.concatenate([selected, extra])

    rng.shuffle(selected)
    return selected


def main():
    print("=" * 72)
    print("PHASE 9.3 — QIEA FEATURE-SELECTION PILOT")
    print("=" * 72)

    bundle = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )
    data = bundle["data"]

    manifest = pd.read_csv(MANIFEST_FILE)
    feature_names = manifest["feature_name"].astype(str).tolist()

    X = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)

    train_indices = torch.where(data.train_mask)[0].numpy()
    val_indices = torch.where(data.val_mask)[0].numpy()

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            "Feature manifest count does not match graph feature width."
        )

    # Fixed, stratified training sample for fast screening.
    train_sample = stratified_indices(
        train_indices, y, TRAIN_SAMPLE_SIZE, SEED
    )

    # Fixed validation sample; never sample from the test split.
    val_sample_size = min(30_000, len(val_indices))
    val_sample = stratified_indices(
        val_indices, y, val_sample_size, SEED + 1
    )

    X_train = X[train_sample]
    y_train = y[train_sample]
    X_val = X[val_sample]
    y_val = y[val_sample]

    print(f"Features: {X.shape[1]}")
    print(f"Training sample: {len(train_sample):,}")
    print(f"Validation sample: {len(val_sample):,}")
    print(f"Training positive rate: {y_train.mean():.4%}")
    print(f"Validation positive rate: {y_val.mean():.4%}")

    # Fit a baseline once using all features.
    all_mask = np.ones(X.shape[1], dtype=np.int8)

    def fit_and_score(mask):
        selected = np.flatnonzero(mask)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=200,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ])

        model.fit(X_train[:, selected], y_train)
        probabilities = model.predict_proba(
            X_val[:, selected]
        )[:, 1]

        pr_auc = average_precision_score(y_val, probabilities)
        predictions = (probabilities >= 0.5).astype(np.int64)
        f2 = fbeta_score(
            y_val, predictions, beta=2, zero_division=0
        )

        return {
            "pr_auc": float(pr_auc),
            "f2": float(f2),
        }

    print("\nEvaluating all-feature screening baseline...")
    baseline = fit_and_score(all_mask)
    print(f"Baseline PR-AUC: {baseline['pr_auc']:.6f}")
    print(f"Baseline F2:     {baseline['f2']:.6f}")

    # Cache is handled by QIEAFeatureSelector.
    selector = QIEAFeatureSelector(
        n_features=X.shape[1],
        evaluator=fit_and_score,
        config=CONFIG,
    )

    print("\nRunning QIEA...")
    result = selector.run()

    selected_indices = result["selected_indices"]
    selected_names = [
        feature_names[int(i)] for i in selected_indices
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_df = pd.DataFrame({
        "feature_index": selected_indices,
        "feature_name": selected_names,
    })
    selected_df.to_csv(
        OUTPUT_DIR / "qiea_selected_features.csv",
        index=False,
    )

    pd.DataFrame(result["history"]).to_csv(
        OUTPUT_DIR / "qiea_history.csv",
        index=False,
    )

    report = {
        "stage": "screening pilot",
        "model": "class-weighted logistic regression",
        "fitness_objective": "validation PR-AUC",
        "training_sample_size": int(len(train_sample)),
        "validation_sample_size": int(len(val_sample)),
        "all_feature_baseline": baseline,
        "selected_count": int(result["selected_count"]),
        "selected_features": selected_names,
        "selected_indices": selected_indices.tolist(),
        "best_metrics": result["best_metrics"],
        "unique_subsets_evaluated": int(
            result["unique_subsets_evaluated"]
        ),
        "config": {
            "population_size": CONFIG.population_size,
            "generations": CONFIG.generations,
            "min_features": CONFIG.min_features,
            "max_features": CONFIG.max_features,
            "random_seed": CONFIG.random_seed,
        },
        "warning": (
            "Screening proxy only. Do not interpret these results "
            "as final GNN performance. The selected subset must be "
            "retrained and evaluated with the GNN."
        ),
    }

    with (
        OUTPUT_DIR / "qiea_pilot_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 72)
    print("QIEA PILOT RESULTS")
    print("=" * 72)
    print(f"Selected features: {result['selected_count']}")
    print(f"Unique subsets evaluated: {result['unique_subsets_evaluated']}")
    print(f"Best validation PR-AUC: {result['best_metrics']['pr_auc']:.6f}")
    print(f"Best validation F2: {result['best_metrics']['f2']:.6f}")

    print("\nSelected features:")
    for name in selected_names:
        print(f"  - {name}")

    print(f"\nArtifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
