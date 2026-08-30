from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
    / "temporal_sage_model.pt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
    / "temporal_threshold_results.csv"
)

THRESHOLDS = np.arange(
    0.01,
    1.00,
    0.01,
)


def evaluate(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    return {
        "threshold": threshold,
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
        "positive_predictions": int(
            predictions.sum()
        ),
    }


def main():

    print("=" * 75)
    print("PHASE 7F-5 — TEMPORAL THRESHOLD SELECTION")
    print("=" * 75)

    print(
        "\n[1/4] Loading temporal GraphSAGE results..."
    )

    payload = torch_load(
        MODEL_FILE
    )

    val_probabilities = np.asarray(
        payload[
            "validation_probabilities"
        ]
    )

    test_probabilities = np.asarray(
        payload[
            "test_probabilities"
        ]
    )

    # The labels are stored in the temporal experiment.
    temporal_file = (
        PROJECT_ROOT
        / "data"
        / "graphs"
        / "temporal"
        / "temporal_experiment.pt"
    )

    temporal = torch_load(
        temporal_file
    )

    y_val = (
        temporal["validation"]
        .y
        .numpy()
    )

    y_test = (
        temporal["test"]
        .y
        .numpy()
    )

    print(
        f"Validation accounts: "
        f"{len(y_val):,}"
    )

    print(
        f"Test accounts: "
        f"{len(y_test):,}"
    )

    # ---------------------------------------------------------
    # Validation threshold search
    # ---------------------------------------------------------

    print(
        "\n[2/4] Searching validation thresholds..."
    )

    results = []

    for threshold in THRESHOLDS:

        metrics = evaluate(
            y_val,
            val_probabilities,
            threshold,
        )

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_f2 = results_df.loc[
        results_df["f2"].idxmax()
    ]

    print("\nBEST VALIDATION F1")
    print(
        f"Threshold: {best_f1['threshold']:.2f}"
    )
    print(
        f"Precision: {best_f1['precision']:.6f}"
    )
    print(
        f"Recall:    {best_f1['recall']:.6f}"
    )
    print(
        f"F1:        {best_f1['f1']:.6f}"
    )
    print(
        f"F2:        {best_f1['f2']:.6f}"
    )

    print("\nBEST VALIDATION F2")
    print(
        f"Threshold: {best_f2['threshold']:.2f}"
    )
    print(
        f"Precision: {best_f2['precision']:.6f}"
    )
    print(
        f"Recall:    {best_f2['recall']:.6f}"
    )
    print(
        f"F1:        {best_f2['f1']:.6f}"
    )
    print(
        f"F2:        {best_f2['f2']:.6f}"
    )

    # ---------------------------------------------------------
    # Test using frozen validation threshold
    # ---------------------------------------------------------

    print(
        "\n[3/4] Applying validation-selected F2 threshold to test..."
    )

    threshold = float(
        best_f2["threshold"]
    )

    test_metrics = evaluate(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\n" + "=" * 75)
    print("TEMPORAL TEST — GRAPHSAGE")
    print("=" * 75)

    for name, value in test_metrics.items():

        if name == "threshold":

            print(
                f"{name.upper():<22}: "
                f"{value:.2f}"
            )

        elif name == "positive_predictions":

            print(
                f"{name.upper():<22}: "
                f"{value:,}"
            )

        else:

            print(
                f"{name.upper():<22}: "
                f"{value:.6f}"
            )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print(
        "\n[4/4] Saving threshold results..."
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("=" * 75)


def torch_load(path):

    import torch

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


if __name__ == "__main__":
    main()