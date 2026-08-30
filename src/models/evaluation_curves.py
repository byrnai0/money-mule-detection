from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "tuned_voting_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
)

PR_CSV = (
    OUTPUT_DIR
    / "precision_recall_curve.csv"
)

ROC_CSV = (
    OUTPUT_DIR
    / "roc_curve.csv"
)

PR_PLOT = (
    OUTPUT_DIR
    / "precision_recall_curve.png"
)

ROC_PLOT = (
    OUTPUT_DIR
    / "roc_curve.png"
)

OPERATING_THRESHOLD = 0.79


def main() -> None:

    print("=" * 75)
    print("PHASE 7C — PRECISION-RECALL AND ROC CURVES")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load test predictions
    # ---------------------------------------------------------

    print("\n[1/5] Loading test predictions...")

    df = pd.read_csv(
        PREDICTION_FILE,
        usecols=[
            "account_id",
            "is_illicit",
            "tuned_vote_test_prob",
        ],
    )

    df = df[
        df["tuned_vote_test_prob"].notna()
    ].copy()

    y_true = df[
        "is_illicit"
    ].to_numpy(
        dtype=np.int8
    )

    probabilities = df[
        "tuned_vote_test_prob"
    ].to_numpy(
        dtype=np.float64
    )

    print(
        f"Test accounts: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Calculate PR curve
    # ---------------------------------------------------------

    print(
        "\n[2/5] Calculating Precision-Recall curve..."
    )

    precision, recall, pr_thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    pr_curve = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
        }
    )

    # sklearn has one fewer threshold than precision/recall.
    threshold_column = np.full(
        len(precision),
        np.nan,
    )

    threshold_column[
        :len(pr_thresholds)
    ] = pr_thresholds

    pr_curve[
        "threshold"
    ] = threshold_column

    pr_curve.to_csv(
        PR_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # Find nearest threshold on PR curve
    # ---------------------------------------------------------

    if len(pr_thresholds) > 0:

        nearest_index = int(
            np.argmin(
                np.abs(
                    pr_thresholds
                    - OPERATING_THRESHOLD
                )
            )
        )

        operating_precision = precision[
            nearest_index
        ]

        operating_recall = recall[
            nearest_index
        ]

    else:

        operating_precision = np.nan
        operating_recall = np.nan

    # ---------------------------------------------------------
    # Plot PR curve
    # ---------------------------------------------------------

    print(
        "\n[3/5] Creating Precision-Recall plot..."
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"Tuned Voting (AP = {pr_auc:.4f})",
    )

    if not np.isnan(
        operating_precision
    ):

        plt.scatter(
            operating_recall,
            operating_precision,
            label=(
                f"Threshold ≈ {OPERATING_THRESHOLD:.2f}"
            ),
        )

    prevalence = y_true.mean()

    plt.axhline(
        prevalence,
        linestyle="--",
        label=(
            f"Random baseline = {prevalence:.4f}"
        ),
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve — Tuned Soft Voting"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        PR_PLOT,
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # ROC curve
    # ---------------------------------------------------------

    print(
        "\n[4/5] Calculating ROC curve..."
    )

    fpr, tpr, roc_thresholds = (
        roc_curve(
            y_true,
            probabilities,
        )
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    roc_curve_df = pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": roc_thresholds,
        }
    )

    roc_curve_df.to_csv(
        ROC_CSV,
        index=False,
    )

    print(
        "Creating ROC plot..."
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"Tuned Voting (AUC = {roc_auc:.4f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve — Tuned Soft Voting"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        ROC_PLOT,
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print(
        "\n[5/5] Final curve summary..."
    )

    print("\n" + "=" * 75)
    print("CURVE RESULTS")
    print("=" * 75)

    print(
        f"\nPR-AUC:   {pr_auc:.6f}"
    )

    print(
        f"ROC-AUC:  {roc_auc:.6f}"
    )

    print(
        f"Positive prevalence: "
        f"{prevalence:.6%}"
    )

    print(
        f"Operating threshold: "
        f"{OPERATING_THRESHOLD:.2f}"
    )

    if not np.isnan(
        operating_precision
    ):

        print(
            f"PR precision near threshold: "
            f"{operating_precision:.6f}"
        )

        print(
            f"PR recall near threshold:    "
            f"{operating_recall:.6f}"
        )

    print(
        f"\nPR curve data:\n{PR_CSV}"
    )

    print(
        f"PR plot:\n{PR_PLOT}"
    )

    print(
        f"\nROC curve data:\n{ROC_CSV}"
    )

    print(
        f"ROC plot:\n{ROC_PLOT}"
    )

    print("\n" + "=" * 75)
    print("PHASE 7C COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()