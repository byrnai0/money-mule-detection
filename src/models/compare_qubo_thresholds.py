
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


ENSEMBLES = {
    "existing_tuned_voting": np.array(
        [0.78, 0.19, 0.03],
        dtype=float,
    ),
    "qubo": np.array(
        [0.68, 0.32, 0.00],
        dtype=float,
    ),
    "classical_brier_grid": np.array(
        [0.66, 0.34, 0.00],
        dtype=float,
    ),
}


# A 0.001 grid gives 981 candidate thresholds.
THRESHOLDS = np.linspace(0.01, 0.99, 981)


def make_probabilities(df, split, weights):
    columns = [
        f"{name}_{split}_prob"
        for name in MODEL_NAMES
    ]

    mask = df[columns].notna().all(axis=1)
    subset = df.loc[mask]

    X = subset[columns].to_numpy(dtype=float)
    y = subset["is_illicit"].to_numpy(dtype=np.int64)

    if len(y) == 0:
        raise ValueError(f"No {split} predictions found.")

    if not np.isfinite(X).all():
        raise ValueError(f"Invalid {split} prediction values.")

    if np.any((X < 0) | (X > 1)):
        raise ValueError(f"{split} probabilities must be in [0, 1].")

    return y, X @ weights


def threshold_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "f2": float(
            fbeta_score(
                y_true, predictions, beta=2, zero_division=0
            )
        ),
        "flagged_count": int(predictions.sum()),
    }


def ranking_metrics(y_true, probabilities):
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
    }


def select_f2_threshold(y_true, probabilities):
    best_result = None

    for threshold in THRESHOLDS:
        result = threshold_metrics(
            y_true,
            probabilities,
            threshold,
        )

        # Maximize F2. For an exact tie, prefer higher precision.
        key = (result["f2"], result["precision"])

        if best_result is None or key > best_result["key"]:
            best_result = {
                "key": key,
                "metrics": result,
            }

    return best_result["metrics"]


def main():
    print("=" * 76)
    print("PHASE 9 — VALIDATION THRESHOLD COMPARISON")
    print("=" * 76)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    results = {}

    # --------------------------------------------------------
    # Tune thresholds using validation data only
    # --------------------------------------------------------

    print("\n[1/3] Selecting thresholds on validation data...")

    validation_data = {}

    for name, weights in ENSEMBLES.items():
        y_val, p_val = make_probabilities(
            df, "val", weights
        )

        selected = select_f2_threshold(y_val, p_val)

        validation_data[name] = {
            "y": y_val,
            "probabilities": p_val,
            "threshold": selected["threshold"],
        }

        results[name] = {
            "weights": {
                model: float(weight)
                for model, weight in zip(MODEL_NAMES, weights)
            },
            "validation": {
                **ranking_metrics(y_val, p_val),
                **selected,
            },
        }

        print(f"\n{name}")
        print(f"  Weights: {weights.tolist()}")
        print(f"  Selected threshold: {selected['threshold']:.3f}")
        print(f"  Validation F2: {selected['f2']:.6f}")
        print(f"  Validation precision: {selected['precision']:.6f}")
        print(f"  Validation recall: {selected['recall']:.6f}")

    # --------------------------------------------------------
    # Evaluate test set with frozen validation thresholds
    # --------------------------------------------------------

    print("\n[2/3] Evaluating held-out test data...")

    for name, weights in ENSEMBLES.items():
        y_test, p_test = make_probabilities(
            df, "test", weights
        )

        frozen_threshold = validation_data[name]["threshold"]

        results[name]["test"] = {
            **ranking_metrics(y_test, p_test),
            **threshold_metrics(
                y_test,
                p_test,
                frozen_threshold,
            ),
        }

        print(f"\n{name}")
        for metric, value in results[name]["test"].items():
            print(f"  {metric:<18}: {value}")

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    print("\n[3/3] Saving report...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "threshold_selection_metric": "validation F2",
        "threshold_grid": {
            "minimum": float(THRESHOLDS.min()),
            "maximum": float(THRESHOLDS.max()),
            "step": float(THRESHOLDS[1] - THRESHOLDS[0]),
        },
        "test_set_used_for_threshold_selection": False,
        "ensembles": results,
        "interpretation": (
            "Thresholds were selected independently on validation data "
            "and then frozen for test evaluation. PR-AUC and ROC-AUC "
            "are threshold-independent. The QUBO weights were fixed "
            "before this comparison."
        ),
    }

    report_file = OUTPUT_DIR / "threshold_comparison_report.json"

    with report_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    summary_rows = []

    for name, result in results.items():
        for split in ("validation", "test"):
            row = {
                "ensemble": name,
                "split": split,
                **result[split],
            }
            summary_rows.append(row)

    summary_file = OUTPUT_DIR / "threshold_comparison.csv"

    pd.DataFrame(summary_rows).to_csv(
        summary_file,
        index=False,
    )

    print(f"\nSaved report: {report_file}")
    print(f"Saved CSV:    {summary_file}")
    print("\nThreshold comparison complete.")


if __name__ == "__main__":
    main()
