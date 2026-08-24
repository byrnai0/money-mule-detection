from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "ensemble_voting_results.csv"
)


MODELS = [
    "sage",
    "gatv2",
    "gin",
]


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    metrics = {
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
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
    }

    return metrics


def main():

    print("=" * 75)
    print("PHASE 6.2 — EQUAL-WEIGHT SOFT VOTING")
    print("=" * 75)

    print("\n[1/4] Loading prediction table...")

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "account_id",
        "is_illicit",
        "sage_val_prob",
        "gatv2_val_prob",
        "gin_val_prob",
        "sage_test_prob",
        "gatv2_test_prob",
        "gin_test_prob",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    # ---------------------------------------------------------
    # Validation set
    # ---------------------------------------------------------

    print(
        "\n[2/4] Computing equal-weight validation ensemble..."
    )

    val_mask = (
        df["sage_val_prob"].notna()
        & df["gatv2_val_prob"].notna()
        & df["gin_val_prob"].notna()
    )

    val_df = df.loc[
        val_mask
    ].copy()

    val_probability = (
        val_df["sage_val_prob"].to_numpy()
        + val_df["gatv2_val_prob"].to_numpy()
        + val_df["gin_val_prob"].to_numpy()
    ) / 3.0

    val_labels = (
        val_df["is_illicit"].to_numpy(
            dtype=np.int64
        )
    )

    val_metrics = evaluate(
        val_labels,
        val_probability,
    )

    # ---------------------------------------------------------
    # Test set
    # ---------------------------------------------------------

    print(
        "\n[3/4] Computing equal-weight test ensemble..."
    )

    test_mask = (
        df["sage_test_prob"].notna()
        & df["gatv2_test_prob"].notna()
        & df["gin_test_prob"].notna()
    )

    test_df = df.loc[
        test_mask
    ].copy()

    test_probability = (
        test_df["sage_test_prob"].to_numpy()
        + test_df["gatv2_test_prob"].to_numpy()
        + test_df["gin_test_prob"].to_numpy()
    ) / 3.0

    test_labels = (
        test_df["is_illicit"].to_numpy(
            dtype=np.int64
        )
    )

    test_metrics = evaluate(
        test_labels,
        test_probability,
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("VALIDATION RESULTS")
    print("=" * 75)

    for name, value in val_metrics.items():
        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    print("\n" + "=" * 75)
    print("TEST RESULTS")
    print("=" * 75)

    for name, value in test_metrics.items():
        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    # ---------------------------------------------------------
    # Save per-account ensemble probabilities
    # ---------------------------------------------------------

    print(
        "\n[4/4] Saving ensemble probabilities..."
    )

    output = df[
        [
            "account_id",
            "is_illicit",
        ]
    ].copy()

    output["sage_val_prob"] = df[
        "sage_val_prob"
    ]

    output["gatv2_val_prob"] = df[
        "gatv2_val_prob"
    ]

    output["gin_val_prob"] = df[
        "gin_val_prob"
    ]

    output["sage_test_prob"] = df[
        "sage_test_prob"
    ]

    output["gatv2_test_prob"] = df[
        "gatv2_test_prob"
    ]

    output["gin_test_prob"] = df[
        "gin_test_prob"
    ]

    output["equal_vote_val_prob"] = np.nan
    output["equal_vote_test_prob"] = np.nan

    output.loc[
        val_mask,
        "equal_vote_val_prob",
    ] = val_probability

    output.loc[
        test_mask,
        "equal_vote_test_prob",
    ] = test_probability

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()