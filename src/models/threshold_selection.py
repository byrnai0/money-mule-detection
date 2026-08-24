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

PREDICTION_FILE = (
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

STACKING_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "stacking_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "threshold_analysis.csv"
)

FINAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "final_detector_results.csv"
)


THRESHOLDS = np.arange(
    0.01,
    1.00,
    0.01,
)


def metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

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


def find_best(
    results: pd.DataFrame,
    metric: str,
) -> pd.Series:

    index = results[metric].idxmax()

    return results.loc[index]


def evaluate_operating_point(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

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


def main() -> None:

    print("=" * 75)
    print("PHASE 6.5 — THRESHOLD SELECTION")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load prediction sources
    # ---------------------------------------------------------

    print("\n[1/5] Loading prediction files...")

    base = pd.read_csv(
        PREDICTION_FILE
    )

    tuned = pd.read_csv(
        TUNED_VOTING_FILE,
        usecols=[
            "account_id",
            "tuned_vote_val_prob",
            "tuned_vote_test_prob",
        ],
    )

    stacking = pd.read_csv(
        STACKING_FILE,
        usecols=[
            "account_id",
            "stacking_val_prob",
            "stacking_test_prob",
        ],
    )

    df = base.merge(
        tuned,
        on="account_id",
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        stacking,
        on="account_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Validation rows
    # ---------------------------------------------------------

    print("\n[2/5] Preparing validation data...")

    val_mask = (
        df["tuned_vote_val_prob"].notna()
        & df["stacking_val_prob"].notna()
    )

    val = df.loc[
        val_mask
    ].copy()

    y_val = val[
        "is_illicit"
    ].to_numpy(
        dtype=np.int64
    )

    tuned_val = val[
        "tuned_vote_val_prob"
    ].to_numpy()

    stacking_val = val[
        "stacking_val_prob"
    ].to_numpy()

    print(
        f"Validation accounts: {len(val):,}"
    )

    # ---------------------------------------------------------
    # Threshold sweep
    # ---------------------------------------------------------

    print(
        "\n[3/5] Searching thresholds..."
    )

    rows = []

    for threshold in THRESHOLDS:

        tuned_metrics = metrics_at_threshold(
            y_val,
            tuned_val,
            threshold,
        )

        stacking_metrics = metrics_at_threshold(
            y_val,
            stacking_val,
            threshold,
        )

        rows.append(
            {
                "threshold": threshold,

                "tuned_precision":
                    tuned_metrics["precision"],

                "tuned_recall":
                    tuned_metrics["recall"],

                "tuned_f1":
                    tuned_metrics["f1"],

                "tuned_f2":
                    tuned_metrics["f2"],

                "tuned_positive_predictions":
                    tuned_metrics["positive_predictions"],

                "stacking_precision":
                    stacking_metrics["precision"],

                "stacking_recall":
                    stacking_metrics["recall"],

                "stacking_f1":
                    stacking_metrics["f1"],

                "stacking_f2":
                    stacking_metrics["f2"],

                "stacking_positive_predictions":
                    stacking_metrics["positive_predictions"],
            }
        )

    analysis = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------
    # Best thresholds
    # ---------------------------------------------------------

    print(
        "\n[4/5] Selecting operating thresholds..."
    )

    best_tuned_f1 = find_best(
        analysis,
        "tuned_f1",
    )

    best_tuned_f2 = find_best(
        analysis,
        "tuned_f2",
    )

    best_stacking_f1 = find_best(
        analysis,
        "stacking_f1",
    )

    best_stacking_f2 = find_best(
        analysis,
        "stacking_f2",
    )

    print("\nTUNED VOTING — BEST F1")
    print(
        f"Threshold: {best_tuned_f1['threshold']:.2f}"
    )
    print(
        f"Precision: {best_tuned_f1['tuned_precision']:.6f}"
    )
    print(
        f"Recall:    {best_tuned_f1['tuned_recall']:.6f}"
    )
    print(
        f"F1:        {best_tuned_f1['tuned_f1']:.6f}"
    )
    print(
        f"F2:        {best_tuned_f1['tuned_f2']:.6f}"
    )

    print("\nTUNED VOTING — BEST F2")
    print(
        f"Threshold: {best_tuned_f2['threshold']:.2f}"
    )
    print(
        f"Precision: {best_tuned_f2['tuned_precision']:.6f}"
    )
    print(
        f"Recall:    {best_tuned_f2['tuned_recall']:.6f}"
    )
    print(
        f"F1:        {best_tuned_f2['tuned_f1']:.6f}"
    )
    print(
        f"F2:        {best_tuned_f2['tuned_f2']:.6f}"
    )

    print("\nSTACKING — BEST F1")
    print(
        f"Threshold: {best_stacking_f1['threshold']:.2f}"
    )
    print(
        f"Precision: {best_stacking_f1['stacking_precision']:.6f}"
    )
    print(
        f"Recall:    {best_stacking_f1['stacking_recall']:.6f}"
    )
    print(
        f"F1:        {best_stacking_f1['stacking_f1']:.6f}"
    )
    print(
        f"F2:        {best_stacking_f1['stacking_f2']:.6f}"
    )

    print("\nSTACKING — BEST F2")
    print(
        f"Threshold: {best_stacking_f2['threshold']:.2f}"
    )
    print(
        f"Precision: {best_stacking_f2['stacking_precision']:.6f}"
    )
    print(
        f"Recall:    {best_stacking_f2['stacking_recall']:.6f}"
    )
    print(
        f"F1:        {best_stacking_f2['stacking_f1']:.6f}"
    )
    print(
        f"F2:        {best_stacking_f2['stacking_f2']:.6f}"
    )

    # ---------------------------------------------------------
    # Apply frozen F2 threshold to TEST
    # ---------------------------------------------------------

    print(
        "\n[5/5] Applying frozen validation thresholds to test..."
    )

    test_mask = (
        df["tuned_vote_test_prob"].notna()
        & df["stacking_test_prob"].notna()
    )

    test = df.loc[
        test_mask
    ].copy()

    y_test = test[
        "is_illicit"
    ].to_numpy(
        dtype=np.int64
    )

    tuned_test = test[
        "tuned_vote_test_prob"
    ].to_numpy()

    stacking_test = test[
        "stacking_test_prob"
    ].to_numpy()

    tuned_f2_threshold = float(
        best_tuned_f2["threshold"]
    )

    stacking_f2_threshold = float(
        best_stacking_f2["threshold"]
    )

    tuned_test_metrics = evaluate_operating_point(
        y_test,
        tuned_test,
        tuned_f2_threshold,
    )

    stacking_test_metrics = evaluate_operating_point(
        y_test,
        stacking_test,
        stacking_f2_threshold,
    )

    print("\n" + "=" * 75)
    print("FINAL TEST OPERATING POINTS — F2 THRESHOLDS")
    print("=" * 75)

    print("\nTUNED VOTING")

    for name, value in tuned_test_metrics.items():

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

    print("\nSTACKING")

    for name, value in stacking_test_metrics.items():

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
    # Save complete threshold analysis
    # ---------------------------------------------------------

    analysis.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Save final per-account results
    # ---------------------------------------------------------

    final = df[
        [
            "account_id",
            "is_illicit",
            "tuned_vote_test_prob",
            "stacking_test_prob",
        ]
    ].copy()

    final["tuned_vote_prediction"] = (
        final["tuned_vote_test_prob"]
        >= tuned_f2_threshold
    ).astype("int8")

    final["stacking_prediction"] = (
        final["stacking_test_prob"]
        >= stacking_f2_threshold
    ).astype("int8")

    FINAL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        FINAL_FILE,
        index=False,
    )

    print(
        f"\nThreshold analysis saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Final predictions saved to:\n"
        f"{FINAL_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()