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
    / "tuned_voting_results.csv"
)

WEIGHT_STEP = 0.01


def evaluate(
    y_true,
    probabilities,
    threshold=0.50,
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
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


def main():

    print("=" * 75)
    print("PHASE 6.3 — TUNED SOFT VOTING")
    print("=" * 75)

    df = pd.read_csv(INPUT_FILE)

    # ---------------------------------------------------------
    # Validation data
    # ---------------------------------------------------------

    val_mask = (
        df["sage_val_prob"].notna()
        & df["gatv2_val_prob"].notna()
        & df["gin_val_prob"].notna()
    )

    val_df = df.loc[
        val_mask
    ].copy()

    sage_val = val_df[
        "sage_val_prob"
    ].to_numpy()

    gatv2_val = val_df[
        "gatv2_val_prob"
    ].to_numpy()

    gin_val = val_df[
        "gin_val_prob"
    ].to_numpy()

    y_val = val_df[
        "is_illicit"
    ].to_numpy(
        dtype=np.int64
    )

    # ---------------------------------------------------------
    # Search weights using VALIDATION ONLY
    # ---------------------------------------------------------

    print(
        "\nSearching validation weights..."
    )

    best_score = -float("inf")
    best_weights = None
    best_metrics = None

    steps = int(
        round(1.0 / WEIGHT_STEP)
    )

    combinations_tested = 0

    for i in range(
        steps + 1
    ):

        w_sage = i * WEIGHT_STEP

        for j in range(
            steps - i + 1
        ):

            w_gatv2 = j * WEIGHT_STEP
            w_gin = (
                1.0
                - w_sage
                - w_gatv2
            )

            probabilities = (
                w_sage * sage_val
                + w_gatv2 * gatv2_val
                + w_gin * gin_val
            )

            metrics = evaluate(
                y_val,
                probabilities,
            )

            # PRIMARY OBJECTIVE:
            # PR-AUC
            score = metrics["pr_auc"]

            combinations_tested += 1

            if score > best_score:

                best_score = score
                best_weights = (
                    w_sage,
                    w_gatv2,
                    w_gin,
                )
                best_metrics = metrics

    print(
        f"Weight combinations tested: "
        f"{combinations_tested:,}"
    )

    print("\nBest validation weights:")

    print(
        f"GraphSAGE: {best_weights[0]:.2f}"
    )

    print(
        f"GATv2:     {best_weights[1]:.2f}"
    )

    print(
        f"GIN:       {best_weights[2]:.2f}"
    )

    print("\nBest validation results:")

    for name, value in best_metrics.items():

        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    # ---------------------------------------------------------
    # Apply weights to TEST only after weights are frozen
    # ---------------------------------------------------------

    test_mask = (
        df["sage_test_prob"].notna()
        & df["gatv2_test_prob"].notna()
        & df["gin_test_prob"].notna()
    )

    test_df = df.loc[
        test_mask
    ].copy()

    sage_test = test_df[
        "sage_test_prob"
    ].to_numpy()

    gatv2_test = test_df[
        "gatv2_test_prob"
    ].to_numpy()

    gin_test = test_df[
        "gin_test_prob"
    ].to_numpy()

    y_test = test_df[
        "is_illicit"
    ].to_numpy(
        dtype=np.int64
    )

    tuned_test_prob = (
        best_weights[0] * sage_test
        + best_weights[1] * gatv2_test
        + best_weights[2] * gin_test
    )

    test_metrics = evaluate(
        y_test,
        tuned_test_prob,
    )

    print("\n" + "=" * 75)
    print("TUNED VOTING TEST RESULTS")
    print("=" * 75)

    for name, value in test_metrics.items():

        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = df[
        [
            "account_id",
            "is_illicit",
        ]
    ].copy()

    output["tuned_vote_val_prob"] = np.nan
    output["tuned_vote_test_prob"] = np.nan

    val_probability = (
        best_weights[0] * sage_val
        + best_weights[1] * gatv2_val
        + best_weights[2] * gin_val
    )

    output.loc[
        val_mask,
        "tuned_vote_val_prob",
    ] = val_probability

    output.loc[
        test_mask,
        "tuned_vote_test_prob",
    ] = tuned_test_prob

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()