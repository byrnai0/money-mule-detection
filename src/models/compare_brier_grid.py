
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "ensemble_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
    / "phase9_qubo"
)

MODEL_NAMES = ["sage", "gatv2", "gin"]
WEIGHT_STEP = 0.01

# QUBO weights from the initial experiment.
QUBO_WEIGHTS = np.array([0.68, 0.32, 0.00])

# Existing Phase 6 tuned-voting weights.
BASELINE_WEIGHTS = np.array([0.78, 0.19, 0.03])


def load_split(df, split):
    columns = [
        f"{name}_{split}_prob"
        for name in MODEL_NAMES
    ]

    mask = df[columns].notna().all(axis=1)
    subset = df.loc[mask]

    X = subset[columns].to_numpy(dtype=float)
    y = subset["is_illicit"].to_numpy(dtype=int)

    if len(y) == 0:
        raise ValueError(f"No predictions found for {split}.")

    if not np.isfinite(X).all():
        raise ValueError(f"Invalid prediction values in {split}.")

    return X, y


def evaluate(y_true, probabilities, threshold=0.50):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f2": float(
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                zero_division=0,
            )
        ),
    }


def find_best_brier_weights(X, y):
    best_score = float("inf")
    best_weights = None
    best_metrics = None
    combinations = 0

    steps = int(round(1.0 / WEIGHT_STEP))

    for i in range(steps + 1):
        w_sage = i * WEIGHT_STEP

        for j in range(steps - i + 1):
            w_gatv2 = j * WEIGHT_STEP
            w_gin = round(
                1.0 - w_sage - w_gatv2,
                10,
            )

            weights = np.array([
                w_sage,
                w_gatv2,
                w_gin,
            ])

            probabilities = X @ weights

            score = brier_score_loss(y, probabilities)
            combinations += 1

            if score < best_score:
                best_score = score
                best_weights = weights.copy()
                best_metrics = evaluate(y, probabilities)

    return {
        "weights": best_weights,
        "brier_score": best_score,
        "metrics": best_metrics,
        "combinations_tested": combinations,
    }


def main():
    print("=" * 76)
    print("PHASE 9 — QUBO VS EXHAUSTIVE BRIER SEARCH")
    print("=" * 76)

    df = pd.read_csv(INPUT_FILE)

    X_val, y_val = load_split(df, "val")
    X_test, y_test = load_split(df, "test")

    print(f"\nValidation accounts: {len(y_val):,}")
    print(f"Test accounts:       {len(y_test):,}")

    # --------------------------------------------------------
    # Optimize weights using validation data only
    # --------------------------------------------------------

    print("\nSearching all valid weight combinations...")

    classical = find_best_brier_weights(X_val, y_val)

    configurations = {
        "classical_brier_grid": classical["weights"],
        "qubo_initial": QUBO_WEIGHTS,
        "existing_tuned_voting": BASELINE_WEIGHTS,
    }

    print(
        f"Combinations tested: "
        f"{classical['combinations_tested']:,}"
    )

    print("\nSelected weights:")
    for name, weights in configurations.items():
        print(
            f"  {name:<24} "
            f"{weights[0]:.2f} / "
            f"{weights[1]:.2f} / "
            f"{weights[2]:.2f}"
        )

    # --------------------------------------------------------
    # Compare validation and test performance
    # --------------------------------------------------------

    report = {
        "optimization_objective": "Brier score",
        "weight_step": WEIGHT_STEP,
        "classical_combinations_tested": (
            classical["combinations_tested"]
        ),
        "test_used_for_weight_selection": False,
        "results": {},
    }

    for name, weights in configurations.items():
        val_probabilities = X_val @ weights
        test_probabilities = X_test @ weights

        report["results"][name] = {
            "weights": {
                model: float(weight)
                for model, weight in zip(
                    MODEL_NAMES, weights
                )
            },
            "validation": evaluate(
                y_val,
                val_probabilities,
            ),
            "test": evaluate(
                y_test,
                test_probabilities,
            ),
        }

    print("\n" + "=" * 76)
    print("VALIDATION RESULTS")
    print("=" * 76)

    for name, result in report["results"].items():
        print(f"\n{name}")
        for metric, value in result["validation"].items():
            print(f"  {metric:<16}: {value:.6f}")

    print("\n" + "=" * 76)
    print("HELD-OUT TEST RESULTS")
    print("=" * 76)

    for name, result in report["results"].items():
        print(f"\n{name}")
        for metric, value in result["test"].items():
            print(f"  {metric:<16}: {value:.6f}")

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_file = OUTPUT_DIR / "brier_grid_comparison.json"

    with json_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    rows = []

    for name, result in report["results"].items():
        for split in ("validation", "test"):
            rows.append({
                "ensemble": name,
                "split": split,
                **result[split],
                **{
                    f"{model}_weight": result["weights"][model]
                    for model in MODEL_NAMES
                },
            })

    csv_file = OUTPUT_DIR / "brier_grid_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)

    print(f"\nSaved JSON: {json_file}")
    print(f"Saved CSV:  {csv_file}")
    print("\nClassical comparison complete.")


if __name__ == "__main__":
    main()
