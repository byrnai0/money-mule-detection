from __future__ import annotations

from pathlib import Path

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "deep_error_analysis.csv"
)

THRESHOLD = 0.79

TOP_N = 25


def main():

    print("=" * 75)
    print("PHASE 7E — DEEP ERROR ANALYSIS")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    print("\n[1/5] Loading features...")

    features = pd.read_csv(
        FEATURE_FILE
    )

    print("[2/5] Loading labels...")

    labels = pd.read_csv(
        LABEL_FILE,
        usecols=[
            "account_id",
            "is_illicit",
        ],
    )

    print("[3/5] Loading predictions...")

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

    print("\n[4/5] Preparing test accounts...")

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

    df = df[
        df["tuned_vote_test_prob"].notna()
    ].copy()

    df["prediction"] = (
        df["tuned_vote_test_prob"]
        >= THRESHOLD
    ).astype("int8")

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
    # Important features to show
    # ---------------------------------------------------------

    display_features = [
        "log1p_in_transaction_count",
        "log1p_out_transaction_count",
        "log1p_unique_senders",
        "log1p_unique_receivers",
        "reciprocal_counterparty_count",
        "reciprocity_ratio",
        "counterparty_concentration",
        "activity_burst_ratio",
        "active_days",
        "unique_sender_banks",
        "unique_receiver_banks",
        "incoming_payment_format_diversity",
        "outgoing_payment_format_diversity",
        "receiving_currency_diversity",
        "payment_currency_diversity",
        "incoming_log_amount_mean",
        "incoming_log_amount_max",
        "outgoing_log_amount_mean",
        "outgoing_log_amount_max",
    ]

    display_features = [
        feature
        for feature in display_features
        if feature in df.columns
    ]

    # ---------------------------------------------------------
    # Extract important cases
    # ---------------------------------------------------------

    highest_risk_fp = (
        df[df["error_type"] == "FP"]
        .sort_values(
            "tuned_vote_test_prob",
            ascending=False,
        )
        .head(TOP_N)
        .copy()
    )

    lowest_risk_fn = (
        df[df["error_type"] == "FN"]
        .sort_values(
            "tuned_vote_test_prob",
            ascending=True,
        )
        .head(TOP_N)
        .copy()
    )

    highest_risk_tp = (
        df[df["error_type"] == "TP"]
        .sort_values(
            "tuned_vote_test_prob",
            ascending=False,
        )
        .head(TOP_N)
        .copy()
    )

    lowest_risk_tp = (
        df[df["error_type"] == "TP"]
        .sort_values(
            "tuned_vote_test_prob",
            ascending=True,
        )
        .head(TOP_N)
        .copy()
    )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    selected = pd.concat(
        [
            highest_risk_fp.assign(
                case_type="highest_risk_fp"
            ),
            lowest_risk_fn.assign(
                case_type="lowest_risk_fn"
            ),
            highest_risk_tp.assign(
                case_type="highest_risk_tp"
            ),
            lowest_risk_tp.assign(
                case_type="lowest_risk_tp"
            ),
        ],
        ignore_index=True,
    )

    selected = selected[
        [
            "case_type",
            "account_id",
            "is_illicit",
            "prediction",
            "tuned_vote_test_prob",
        ]
        + display_features
    ]

    selected.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print(
        "\n[5/5] Printing representative cases..."
    )

    for title, subset in [
        (
            "HIGHEST-RISK FALSE POSITIVES",
            highest_risk_fp,
        ),
        (
            "LOWEST-RISK FALSE NEGATIVES",
            lowest_risk_fn,
        ),
        (
            "HIGHEST-RISK TRUE POSITIVES",
            highest_risk_tp,
        ),
        (
            "LOWEST-RISK TRUE POSITIVES",
            lowest_risk_tp,
        ),
    ]:

        print("\n" + "=" * 75)
        print(title)
        print("=" * 75)

        if subset.empty:

            print("No accounts in this category.")

            continue

        columns = [
            "account_id",
            "tuned_vote_test_prob",
        ] + display_features

        print(
            subset[
                columns
            ]
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 75)
    print("PHASE 7E COMPLETE")
    print("=" * 75)

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()