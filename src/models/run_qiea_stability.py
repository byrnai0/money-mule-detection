
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

from qiea_feature_selector import QIEAConfig, QIEAFeatureSelector


ROOT = Path(__file__).resolve().parents[2]
GRAPH_FILE = ROOT / "data" / "graphs" / "training_graph.pt"
MANIFEST_FILE = ROOT / "data" / "processed" / "feature_manifest.csv"

BASE_DIR = ROOT / "data" / "graphs" / "evaluation" / "phase9_qiea"
OUTPUT_DIR = BASE_DIR / "stability"

# Keep the data sample fixed. Vary only the QIEA search seed.
DATA_SEED = 42
QIEA_SEEDS = [11, 22, 33]

TRAIN_SAMPLE_SIZE = 100_000
VAL_SAMPLE_SIZE = 30_000


def stratified_indices(indices, labels, size, seed):
    """Sample indices while preserving class proportions."""
    rng = np.random.default_rng(seed)

    if size >= len(indices):
        return indices.copy()

    local_y = labels[indices]
    classes, counts = np.unique(local_y, return_counts=True)

    # Allocate sample counts using largest-remainder rounding.
    exact = size * counts / len(indices)
    allocations = np.floor(exact).astype(int)
    remainder = size - allocations.sum()

    if remainder:
        order = np.argsort(-(exact - allocations))
        allocations[order[:remainder]] += 1

    selected = []

    for cls, count in zip(classes, allocations):
        class_indices = indices[local_y == cls]
        count = min(int(count), len(class_indices))
        if count:
            selected.extend(
                rng.choice(class_indices, size=count, replace=False).tolist()
            )

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def main():
    print("=" * 72)
    print("PHASE 9.3 — QIEA STABILITY EXPERIMENT")
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
            f"Manifest has {len(feature_names)} features, "
            f"but graph has {X.shape[1]}."
        )

    train_sample = stratified_indices(
        train_indices, y, min(TRAIN_SAMPLE_SIZE, len(train_indices)),
        DATA_SEED,
    )
    val_sample = stratified_indices(
        val_indices, y, min(VAL_SAMPLE_SIZE, len(val_indices)),
        DATA_SEED + 1,
    )

    X_train, y_train = X[train_sample], y[train_sample]
    X_val, y_val = X[val_sample], y[val_sample]

    print(f"Features: {X.shape[1]}")
    print(f"Training sample: {len(train_sample):,}")
    print(f"Validation sample: {len(val_sample):,}")
    print(f"Training positive rate: {y_train.mean():.4%}")
    print(f"Validation positive rate: {y_val.mean():.4%}")

    def fit_and_score(mask):
        selected = np.flatnonzero(mask)
        if len(selected) == 0:
            raise ValueError("QIEA produced an empty feature subset.")

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=200,
                solver="lbfgs",
                random_state=DATA_SEED,
            )),
        ])

        model.fit(X_train[:, selected], y_train)
        probabilities = model.predict_proba(X_val[:, selected])[:, 1]

        return {
            "pr_auc": float(
                average_precision_score(y_val, probabilities)
            ),
            "f2": float(fbeta_score(
                y_val,
                (probabilities >= 0.5).astype(np.int64),
                beta=2,
                zero_division=0,
            )),
        }

    # The baseline is identical for every search seed.
    baseline = fit_and_score(np.ones(X.shape[1], dtype=np.int8))
    print(f"\nAll-feature baseline PR-AUC: {baseline['pr_auc']:.6f}")
    print(f"All-feature baseline F2:     {baseline['f2']:.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    selected_sets = {}

    for seed in QIEA_SEEDS:
        print(f"\n{'-' * 72}")
        print(f"Running QIEA with search seed {seed}")
        print(f"{'-' * 72}")

        config = QIEAConfig(
            population_size=10,
            generations=8,
            min_features=5,
            max_features=22,
            rotation_step=0.03 * np.pi,
            random_seed=seed,
        )

        selector = QIEAFeatureSelector(
            n_features=X.shape[1],
            evaluator=fit_and_score,
            config=config,
        )

        result = selector.run()

        indices = [int(i) for i in result["selected_indices"]]
        names = [feature_names[i] for i in indices]
        selected_sets[seed] = set(indices)

        run_dir = OUTPUT_DIR / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "feature_index": indices,
            "feature_name": names,
        }).to_csv(run_dir / "selected_features.csv", index=False)

        pd.DataFrame(result["history"]).to_csv(
            run_dir / "history.csv", index=False
        )

        report = {
            "search_seed": seed,
            "data_seed": DATA_SEED,
            "training_sample_size": len(train_sample),
            "validation_sample_size": len(val_sample),
            "baseline": baseline,
            "selected_count": len(indices),
            "selected_indices": indices,
            "selected_features": names,
            "best_metrics": result["best_metrics"],
            "unique_subsets_evaluated": result["unique_subsets_evaluated"],
            "config": {
                "population_size": config.population_size,
                "generations": config.generations,
                "min_features": config.min_features,
                "max_features": config.max_features,
                "random_seed": seed,
            },
        }

        with (run_dir / "report.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(report, file, indent=2)

        summaries.append({
            "seed": seed,
            "selected_count": len(indices),
            "pr_auc": float(result["best_metrics"]["pr_auc"]),
            "f2": float(result["best_metrics"]["f2"]),
            "unique_subsets": int(result["unique_subsets_evaluated"]),
            "selected_features": " | ".join(names),
        })

        print(f"Selected features: {len(indices)}")
        print(f"Best PR-AUC: {result['best_metrics']['pr_auc']:.6f}")
        print(f"Best F2:     {result['best_metrics']['f2']:.6f}")
        print(f"Unique subsets: {result['unique_subsets_evaluated']}")

    # Compare seeds with the original seed-42 pilot, if available.
    original_file = BASE_DIR / "qiea_selected_features.csv"
    if original_file.exists():
        original_df = pd.read_csv(original_file)
        original_set = set(original_df["feature_index"].astype(int))
        original_names = set(original_df["feature_name"].astype(str))

        for row in summaries:
            seed = int(row["seed"])
            current = selected_sets[seed]
            row["overlap_with_original_seed42"] = len(current & original_set)
            row["jaccard_with_original_seed42"] = (
                len(current & original_set) / len(current | original_set)
                if current | original_set else 1.0
            )

        # Include the original pilot as a reference, not a new run.
        original_report_file = BASE_DIR / "qiea_pilot_report.json"
        if original_report_file.exists():
            with original_report_file.open(encoding="utf-8") as file:
                original_report = json.load(file)

            summaries.insert(0, {
                "seed": 42,
                "selected_count": original_report["selected_count"],
                "pr_auc": original_report["best_metrics"]["pr_auc"],
                "f2": original_report["best_metrics"]["f2"],
                "unique_subsets": original_report["unique_subsets_evaluated"],
                "selected_features": " | ".join(
                    original_report["selected_features"]
                ),
                "overlap_with_original_seed42": len(original_set),
                "jaccard_with_original_seed42": 1.0,
            })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / "stability_summary.csv", index=False)

    # Count how often each feature is selected across the new runs.
    feature_counts = {}
    for indices in selected_sets.values():
        for index in indices:
            feature_counts[index] = feature_counts.get(index, 0) + 1

    frequency_df = pd.DataFrame([
        {
            "feature_index": i,
            "feature_name": feature_names[i],
            "selected_in_new_runs": feature_counts.get(i, 0),
            "total_new_runs": len(QIEA_SEEDS),
        }
        for i in range(len(feature_names))
    ])
    frequency_df = frequency_df.sort_values(
        ["selected_in_new_runs", "feature_name"],
        ascending=[False, True],
    )
    frequency_df.to_csv(
        OUTPUT_DIR / "feature_selection_frequency.csv", index=False
    )

    print("\n" + "=" * 72)
    print("STABILITY SUMMARY")
    print("=" * 72)
    print(summary_df[
        ["seed", "selected_count", "pr_auc", "f2", "unique_subsets"]
    ].to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
