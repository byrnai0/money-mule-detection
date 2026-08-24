from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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
    / "stacking_results.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "stacking_meta_model.joblib"
)


BASE_MODELS = [
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
    print("PHASE 6.4 — STACKING")
    print("=" * 75)

    print("\n[1/5] Loading prediction table...")

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
    # Build validation matrix
    # ---------------------------------------------------------

    print(
        "\n[2/5] Preparing validation meta-features..."
    )

    val_mask = (
        df["sage_val_prob"].notna()
        & df["gatv2_val_prob"].notna()
        & df["gin_val_prob"].notna()
    )

    val_df = df.loc[
        val_mask
    ].copy()

    X_val = val_df[
        [
            "sage_val_prob",
            "gatv2_val_prob",
            "gin_val_prob",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    y_val = val_df[
        "is_illicit"
    ].to_numpy(
        dtype=np.int64
    )

    print(
        f"Validation samples: {len(y_val):,}"
    )

    # ---------------------------------------------------------
    # Build test matrix
    # ---------------------------------------------------------

    print(
        "\n[3/5] Preparing test meta-features..."
    )

    test_mask = (
        df["sage_test_prob"].notna()
        & df["gatv2_test_prob"].notna()
        & df["gin_test_prob"].notna()
    )

    test_df = df.loc[
        test_mask
    ].copy()

    X_test = test_df[
        [
            "sage_test_prob",
            "gatv2_test_prob",
            "gin_test_prob",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    y_test = test_df[
        "is_illicit"
    ].to_numpy(
        dtype=np.int64
    )

    print(
        f"Test samples: {len(y_test):,}"
    )

    # ---------------------------------------------------------
    # Train meta-classifier
    # ---------------------------------------------------------

    print(
        "\n[4/5] Training logistic-regression meta-classifier..."
    )

    meta_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    meta_model.fit(
        X_val,
        y_val,
    )

    # ---------------------------------------------------------
    # Validation / test predictions
    # ---------------------------------------------------------

    val_probability = meta_model.predict_proba(
        X_val
    )[:, 1]

    test_probability = meta_model.predict_proba(
        X_test
    )[:, 1]

    val_metrics = evaluate(
        y_val,
        val_probability,
    )

    test_metrics = evaluate(
        y_test,
        test_probability,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("STACKING VALIDATION RESULTS")
    print("=" * 75)

    for name, value in val_metrics.items():

        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    print("\n" + "=" * 75)
    print("STACKING TEST RESULTS")
    print("=" * 75)

    for name, value in test_metrics.items():

        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    # ---------------------------------------------------------
    # Meta-model coefficients
    # ---------------------------------------------------------

    classifier = (
        meta_model.named_steps[
            "classifier"
        ]
    )

    print("\nMeta-model coefficients:")

    for model_name, coefficient in zip(
        BASE_MODELS,
        classifier.coef_[0],
    ):

        print(
            f"{model_name.upper():<10}: "
            f"{coefficient:.6f}"
        )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    output = df[
        [
            "account_id",
            "is_illicit",
        ]
    ].copy()

    output["stacking_val_prob"] = np.nan
    output["stacking_test_prob"] = np.nan

    output.loc[
        val_mask,
        "stacking_val_prob",
    ] = val_probability

    output.loc[
        test_mask,
        "stacking_test_prob",
    ] = test_probability

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Save meta-model
    # ---------------------------------------------------------

    import joblib

    joblib.dump(
        meta_model,
        MODEL_FILE,
    )

    print(
        f"\nPredictions saved to:\n{OUTPUT_FILE}"
    )

    print(
        f"Meta-model saved to:\n{MODEL_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()