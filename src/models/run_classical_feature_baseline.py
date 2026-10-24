
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
GRAPH_FILE = ROOT / "data/graphs/training_graph.pt"
MANIFEST_FILE = ROOT / "data/processed/feature_manifest.csv"
OUT_DIR = ROOT / "data/graphs/evaluation/phase9_qiea/classical"

DATA_SEED = 42
TRAIN_SIZE = 100_000
VAL_SIZE = 30_000
FEATURE_COUNTS = [10, 12, 13]


def stratified_indices(indices, labels, size, seed):
    rng = np.random.default_rng(seed)

    if size >= len(indices):
        return indices.copy()

    local_y = labels[indices]
    classes, counts = np.unique(local_y, return_counts=True)
    exact = size * counts / len(indices)
    allocations = np.floor(exact).astype(int)
    remainder = size - allocations.sum()

    if remainder:
        order = np.argsort(-(exact - allocations))
        allocations[order[:remainder]] += 1

    selected = []
    for cls, count in zip(classes, allocations):
        group = indices[local_y == cls]
        selected.extend(
            rng.choice(group, size=int(count), replace=False).tolist()
        )

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            solver="lbfgs",
            random_state=DATA_SEED,
        )),
    ])


def score_subset(X_train, y_train, X_val, y_val, indices):
    model = make_model()
    model.fit(X_train[:, indices], y_train)
    probabilities = model.predict_proba(X_val[:, indices])[:, 1]

    return {
        "pr_auc": float(average_precision_score(y_val, probabilities)),
        "f2": float(fbeta_score(
            y_val,
            (probabilities >= 0.5).astype(int),
            beta=2,
            zero_division=0,
        )),
    }


def main():
    print("PHASE 9.3 — CLASSICAL FEATURE-SELECTION BASELINE", flush=True)

    bundle = torch.load(
        GRAPH_FILE, map_location="cpu", weights_only=False
    )
    data = bundle["data"]
    X = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)

    names = pd.read_csv(MANIFEST_FILE)["feature_name"].astype(str).tolist()
    if len(names) != X.shape[1]:
        raise ValueError("Feature manifest and graph feature counts differ.")

    train_ids = torch.where(data.train_mask)[0].numpy()
    val_ids = torch.where(data.val_mask)[0].numpy()

    train_ids = stratified_indices(
        train_ids, y, min(TRAIN_SIZE, len(train_ids)), DATA_SEED
    )
    val_ids = stratified_indices(
        val_ids, y, min(VAL_SIZE, len(val_ids)), DATA_SEED + 1
    )

    X_train, y_train = X[train_ids], y[train_ids]
    X_val, y_val = X[val_ids], y[val_ids]

    print(f"Training sample: {len(train_ids):,}", flush=True)
    print(f"Validation sample: {len(val_ids):,}", flush=True)

    all_indices = np.arange(X.shape[1])

    # Rank features using training data only.
    ranking_model = make_model()
    ranking_model.fit(X_train, y_train)
    coefficients = np.abs(
        ranking_model.named_steps["classifier"].coef_[0]
    )
    ranking = np.argsort(-coefficients)

    results = []
    for k in FEATURE_COUNTS:
        selected = ranking[:k]
        metrics = score_subset(
            X_train, y_train, X_val, y_val, selected
        )

        print(
            f"Top {k:2d} features | "
            f"PR-AUC={metrics['pr_auc']:.6f} | "
            f"F2={metrics['f2']:.6f}",
            flush=True,
        )

        for rank, index in enumerate(selected, start=1):
            results.append({
                "feature_count": k,
                "rank": rank,
                "feature_index": int(index),
                "feature_name": names[index],
                "absolute_coefficient": float(coefficients[index]),
                "pr_auc": metrics["pr_auc"],
                "f2": metrics["f2"],
            })

    # Include all-feature baseline for reference.
    baseline = score_subset(
        X_train, y_train, X_val, y_val, all_indices
    )
    results.append({
        "feature_count": len(all_indices),
        "rank": 0,
        "feature_index": -1,
        "feature_name": "__ALL_FEATURES_BASELINE__",
        "absolute_coefficient": np.nan,
        "pr_auc": baseline["pr_auc"],
        "f2": baseline["f2"],
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(
        OUT_DIR / "coefficient_ranking_results.csv", index=False
    )

    print(f"\nAll-feature PR-AUC: {baseline['pr_auc']:.6f}")
    print(f"All-feature F2:     {baseline['f2']:.6f}")
    print(f"Saved results to: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
