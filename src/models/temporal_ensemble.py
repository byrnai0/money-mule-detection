from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
    / "temporal_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
    / "temporal_ensemble_results.csv"
)

WEIGHT_STEP = 0.01


def main():

    print("=" * 75)
    print("PHASE 7F-4 — TEMPORAL SOFT VOTING")
    print("=" * 75)

    df = pd.read_csv(
        INPUT_FILE
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    val = df[
        df["period"] == "validation"
    ].copy()

    test = df[
        df["period"] == "test"
    ].copy()

    sage_val = val[
        "sage_prob"
    ].to_numpy()

    gatv2_val = val[
        "gatv2_prob"
    ].to_numpy()

    gin_val = val[
        "gin_prob"
    ].to_numpy()

    y_val = val[
        "is_illicit"
    ].to_numpy(
        dtype=np.int8
    )

    sage_test = test[
        "sage_prob"
    ].to_numpy()

    gatv2_test = test[
        "gatv2_prob"
    ].to_numpy()

    gin_test = test[
        "gin_prob"
    ].to_numpy()

    y_test = test[
        "is_illicit"
    ].to_numpy(
        dtype=np.int8
    )

    # ---------------------------------------------------------
    # Weight search
    # ---------------------------------------------------------

    print(
        "\nSearching temporal validation weights..."
    )

    best_pr_auc = -float("inf")
    best_weights = None

    steps = int(
        round(
            1.0 / WEIGHT_STEP
        )
    )

    combinations = 0

    for i in range(
        steps + 1
    ):

        w_sage = (
            i * WEIGHT_STEP
        )

        for j in range(
            steps - i + 1
        ):

            w_gatv2 = (
                j * WEIGHT_STEP
            )

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

            score = average_precision_score(
                y_val,
                probabilities,
            )

            combinations += 1

            if score > best_pr_auc:

                best_pr_auc = score

                best_weights = (
                    w_sage,
                    w_gatv2,
                    w_gin,
                )

    print(
        f"Combinations tested: "
        f"{combinations:,}"
    )

    print("\nBEST TEMPORAL WEIGHTS")

    print(
        f"GraphSAGE: {best_weights[0]:.2f}"
    )

    print(
        f"GATv2:     {best_weights[1]:.2f}"
    )

    print(
        f"GIN:       {best_weights[2]:.2f}"
    )

    print(
        f"Validation PR-AUC: "
        f"{best_pr_auc:.6f}"
    )

    # ---------------------------------------------------------
    # Test ensemble
    # ---------------------------------------------------------

    test_probability = (
        best_weights[0] * sage_test
        + best_weights[1] * gatv2_test
        + best_weights[2] * gin_test
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probability,
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probability,
    )

    print("\n" + "=" * 75)
    print("TEMPORAL ENSEMBLE TEST RESULTS")
    print("=" * 75)

    print(
        f"PR-AUC:   {test_pr_auc:.6f}"
    )

    print(
        f"ROC-AUC:  {test_roc_auc:.6f}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = df[
        [
            "period",
            "account_id",
            "is_illicit",
        ]
    ].copy()

    output["temporal_ensemble_prob"] = np.nan

    validation_probability = (
        best_weights[0] * sage_val
        + best_weights[1] * gatv2_val
        + best_weights[2] * gin_val
    )

    val_indices = (
        output["period"] == "validation"
    )

    test_indices = (
        output["period"] == "test"
    )

    output.loc[
        val_indices,
        "temporal_ensemble_prob",
    ] = validation_probability

    output.loc[
        test_indices,
        "temporal_ensemble_prob",
    ] = test_probability

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