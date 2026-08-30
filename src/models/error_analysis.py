from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_features_final.csv"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_labels.csv"
)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "tuned_voting_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "error_analysis.csv"
)

THRESHOLD = 0.79


def main():

    print("=" * 75)
    print("PHASE 7B — ERROR ANALYSIS")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load features
    # ---------------------------------------------------------

    print("\n[1/5] Loading final features...")

    features = pd.read_csv(
        FEATURE_FILE
    )

    # ---------------------------------------------------------
    # Load labels
    # ---------------------------------------------------------

    print("[2/5] Loading labels...")

    labels = pd.read_csv(
        LABEL_FILE,
        usecols=[
            "account_id",
            "is_illicit",
        ],
    )

    # ---------------------------------------------------------
    # Load predictions
    # ---------------------------------------------------------

    print("[3/5] Loading tuned-voting predictions...")

    predictions = pd.read_csv(
        PREDICTION_FILE,
        usecols=[
            "account_id",
            "tuned_vote_test_prob",
        ],
    )

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------

    print("[4/5] Aligning features, labels and predictions...")

    df = (
        features
        .merge(
            labels,
            on="account_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            predictions,
            on="account_id",
            how="inner",
            validate="one_to_one",
        )
    )

    # Only test accounts have prediction values.
    df = df[
        df["tuned_vote_test_prob"].notna()
    ].copy()

    df["prediction"] = (
        df["tuned_vote_test_prob"]
        >= THRESHOLD
    ).astype("int8")

    # ---------------------------------------------------------
    # Error categories
    # ---------------------------------------------------------

    df["error_type"] = "TN"

    df.loc[
        (df["is_illicit"] == 1)
        & (df["prediction"] == 1),
        "error_type",
    ] = "TP"

    df.loc[
        (df["is_illicit"] == 1)
        & (df["prediction"] == 0),
        "error_type",
    ] = "FN"

    df.loc[
        (df["is_illicit"] == 0)
        & (df["prediction"] == 1),
        "error_type",
    ] = "FP"

    # ---------------------------------------------------------
    # Print counts
    # ---------------------------------------------------------

    print("\n[5/5] Analyzing prediction groups...")

    counts = (
        df["error_type"]
        .value_counts()
        .reindex(
            [
                "TP",
                "FP",
                "FN",
                "TN",
            ],
            fill_value=0,
        )
    )

    print("\nPrediction groups:")

    for group, count in counts.items():
        print(
            f"{group}: {count:,}"
        )

    # ---------------------------------------------------------
    # Feature comparison
    # ---------------------------------------------------------

    feature_columns = [
        col
        for col in features.columns
        if col != "account_id"
    ]

    comparison_rows = []

    for feature in feature_columns:

        for group in [
            "TP",
            "FP",
            "FN",
            "TN",
        ]:

            subset = df[
                df["error_type"] == group
            ][feature]

            comparison_rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "count": len(subset),
                    "mean": subset.mean(),
                    "median": subset.median(),
                    "std": subset.std(),
                }
            )

    comparison = pd.DataFrame(
        comparison_rows
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Quick console comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("ERROR GROUP SUMMARY")
    print("=" * 75)

    for group in [
        "TP",
        "FP",
        "FN",
        "TN",
    ]:

        subset = df[
            df["error_type"] == group
        ]

        print(
            f"\n{group} ({len(subset):,} accounts)"
        )

        for feature in [
            "log1p_in_transaction_count",
            "log1p_out_transaction_count",
            "log1p_unique_senders",
            "log1p_unique_receivers",
            "reciprocity_ratio",
            "activity_burst_ratio",
            "counterparty_concentration",
            "unique_sender_banks",
            "unique_receiver_banks",
            "payment_currency_diversity",
            "receiving_currency_diversity",
        ]:

            if feature in subset.columns:

                print(
                    f"  {feature:<40}"
                    f"mean={subset[feature].mean():.4f} "
                    f"median={subset[feature].median():.4f}"
                )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()