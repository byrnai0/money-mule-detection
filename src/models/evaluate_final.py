from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "ensemble_predictions.csv"
)

TUNED_VOTING_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "tuned_voting_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "final_evaluation.csv"
)

THRESHOLD = 0.79


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total = len(y_true)

    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "positive_predictions": int(
            predictions.sum()
        ),
        "total_accounts": int(total),
        "flagged_percentage": (
            float(
                predictions.sum()
                / total
                * 100
            )
            if total > 0
            else 0.0
        ),
        "positive_prevalence": (
            float(
                y_true.mean() * 100
            )
            if total > 0
            else 0.0
        ),
    }


def main():

    print("=" * 75)
    print("PHASE 7A — FINAL MODEL EVALUATION")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\n[1/4] Loading prediction data...")

    base = pd.read_csv(
        PREDICTIONS_FILE
    )

    tuned = pd.read_csv(
        TUNED_VOTING_FILE,
        usecols=[
            "account_id",
            "tuned_vote_test_prob",
        ],
    )

    df = base.merge(
        tuned,
        on="account_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Test subset
    # ---------------------------------------------------------

    print(
        "\n[2/4] Preparing untouched test set..."
    )

    test_mask = (
        df["tuned_vote_test_prob"].notna()
    )

    test = df.loc[
        test_mask
    ].copy()

    y_test = test[
        "is_illicit"
    ].to_numpy(
        dtype=np.int8
    )

    probabilities = test[
        "tuned_vote_test_prob"
    ].to_numpy(
        dtype=np.float64
    )

    print(
        f"Test accounts: {len(test):,}"
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    print(
        "\n[3/4] Calculating final metrics..."
    )

    metrics = calculate_metrics(
        y_test,
        probabilities,
        THRESHOLD,
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL TEST RESULTS")
    print("=" * 75)

    print(
        f"\nThreshold:          "
        f"{metrics['threshold']:.2f}"
    )

    print(
        f"PR-AUC:             "
        f"{metrics['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC:            "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Precision:           "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall:              "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1:                  "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"F2:                  "
        f"{metrics['f2']:.6f}"
    )

    print("\nConfusion matrix:")

    print(
        f"True Negatives:     "
        f"{metrics['true_negatives']:,}"
    )

    print(
        f"False Positives:    "
        f"{metrics['false_positives']:,}"
    )

    print(
        f"False Negatives:    "
        f"{metrics['false_negatives']:,}"
    )

    print(
        f"True Positives:     "
        f"{metrics['true_positives']:,}"
    )

    print("\nPrediction volume:")

    print(
        f"Flagged accounts:   "
        f"{metrics['positive_predictions']:,}"
    )

    print(
        f"Flagged percentage:  "
        f"{metrics['flagged_percentage']:.4f}%"
    )

    print(
        f"Actual prevalence:   "
        f"{metrics['positive_prevalence']:.4f}%"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print(
        "\n[4/4] Saving evaluation results..."
    )

    results = pd.DataFrame(
        [metrics]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 75)
    print("PHASE 7A COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()