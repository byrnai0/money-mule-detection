
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "training_graph.pt"
)

IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
    / "global_feature_importance.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
    / "phase9_qiea"
)

RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"


def load_data():
    bundle = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )

    data = bundle["data"]

    importance = pd.read_csv(IMPORTANCE_FILE)

    manifest_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "feature_manifest.csv"
    )

    manifest = pd.read_csv(manifest_file)
    feature_names = manifest["feature_name"].astype(str).tolist()

    if len(feature_names) != data.x.shape[1]:
        raise ValueError(
            f"Manifest has {len(feature_names)} names, "
            f"but graph has {data.x.shape[1]} features."
        )

    if len(feature_names) != data.x.shape[1]:
        raise ValueError(
            f"Feature-name count ({len(feature_names)}) "
            f"does not match matrix width ({data.x.shape[1]})."
        )

    # Explicitly keep all splits separate.
    X = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)

    train_mask = data.train_mask.cpu().numpy().astype(bool)
    val_mask = data.val_mask.cpu().numpy().astype(bool)

    return (
        X,
        y,
        train_mask,
        val_mask,
        feature_names,
    )


def evaluate_subset(
    mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Fit a cheap classical screening model using selected features.

    The preprocessing and model are fitted on training nodes only.
    Validation PR-AUC is the QIEA primary objective.
    F2 is reported at a fixed threshold of 0.5 for diagnostics.
    """

    selected = np.flatnonzero(mask)

    if len(selected) == 0:
        raise ValueError("Feature subset cannot be empty.")

    X_train = X[train_mask][:, selected]
    y_train = y[train_mask]

    X_val = X[val_mask][:, selected]
    y_val = y[val_mask]

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight=CLASS_WEIGHT,
                max_iter=300,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    pr_auc = average_precision_score(
        y_val,
        probabilities,
    )

    f2 = fbeta_score(
        y_val,
        predictions,
        beta=2,
        zero_division=0,
    )

    return {
        "pr_auc": float(pr_auc),
        "f2": float(f2),
        "selected_count": int(len(selected)),
    }


def main():
    print("=" * 72)
    print("PHASE 9.3 — QIEA FEATURE EVALUATOR SMOKE TEST")
    print("=" * 72)

    (
        X,
        y,
        train_mask,
        val_mask,
        feature_names,
    ) = load_data()

    print(f"Feature matrix: {X.shape}")
    print(f"Training nodes: {train_mask.sum():,}")
    print(f"Validation nodes: {val_mask.sum():,}")

    # Test the evaluator on all 26 features first.
    all_features = np.ones(
        X.shape[1],
        dtype=np.int8,
    )

    metrics = evaluate_subset(
        all_features,
        X,
        y,
        train_mask,
        val_mask,
        feature_names,
    )

    print("\nAll-feature logistic-regression baseline:")
    print(f"  PR-AUC: {metrics['pr_auc']:.6f}")
    print(f"  F2:     {metrics['f2']:.6f}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame({
        "feature_index": np.arange(len(feature_names)),
        "feature": feature_names,
    }).to_csv(
        OUTPUT_DIR / "feature_mapping.csv",
        index=False,
    )

    print(f"\nSaved feature mapping to: {OUTPUT_DIR}")
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
