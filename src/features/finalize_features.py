from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_features_v3.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_features_final.csv"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_manifest.csv"
)


# ---------------------------------------------------------
# Final candidate features
# ---------------------------------------------------------

FEATURES = [
    # Activity
    "in_transaction_count",
    "out_transaction_count",

    # Network
    "unique_senders",
    "unique_receivers",
    "reciprocal_counterparty_count",
    "reciprocity_ratio",
    "counterparty_concentration",

    # Temporal
    "incoming_active_days",
    "outgoing_active_days",
    "active_days",
    "activity_burst_ratio",

    # Bank / payment / currency diversity
    "unique_sender_banks",
    "unique_receiver_banks",
    "incoming_payment_format_diversity",
    "outgoing_payment_format_diversity",
    "receiving_currency_diversity",
    "payment_currency_diversity",

    # Incoming amount
    "incoming_log_amount_mean",
    "incoming_log_amount_std",
    "incoming_log_amount_max",
    "incoming_currency_abs_z_mean",
    "incoming_currency_abs_z_max",

    # Outgoing amount
    "outgoing_log_amount_mean",
    "outgoing_log_amount_std",
    "outgoing_log_amount_max",
    "outgoing_currency_abs_z_max",
]


# ---------------------------------------------------------
# Features that need log1p transformation
#
# Only count-like features are transformed here.
# Amounts are already log-transformed in v3.
# ---------------------------------------------------------

LOG_FEATURES = [
    "in_transaction_count",
    "out_transaction_count",
    "unique_senders",
    "unique_receivers",
    "unique_sender_banks",
    "unique_receiver_banks",
    "incoming_active_days",
    "outgoing_active_days",
    "active_days",
    "incoming_payment_format_diversity",
    "outgoing_payment_format_diversity",
    "receiving_currency_diversity",
    "payment_currency_diversity",
    "reciprocal_counterparty_count",
]


def main() -> None:

    print("=" * 75)
    print("PHASE 4 — FINAL FEATURE MATRIX")
    print("=" * 75)

    print("\nLoading candidate features...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Rows: {len(df):,}"
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # ---------------------------------------------------------
    # Keep only selected columns.
    # ---------------------------------------------------------

    final_df = df[
        ["account_id"] + FEATURES
    ].copy()

    # ---------------------------------------------------------
    # Handle invalid numerical values.
    # ---------------------------------------------------------

    numeric_features = FEATURES

    final_df[numeric_features] = (
        final_df[numeric_features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # Log transforms.
    #
    # Create transformed versions while preserving the
    # original feature naming in the manifest.
    # ---------------------------------------------------------

    print("\nApplying log1p transformations...")

    for feature in LOG_FEATURES:

        transformed_name = (
            f"log1p_{feature}"
        )

        final_df[transformed_name] = (
            np.log1p(
                final_df[feature].clip(lower=0)
            )
        )

        # Remove raw count version.
        final_df.drop(
            columns=[feature],
            inplace=True,
        )

    # ---------------------------------------------------------
    # Feature manifest.
    # ---------------------------------------------------------

    manifest_rows = []

    for feature in FEATURES:

        if feature in LOG_FEATURES:

            final_name = f"log1p_{feature}"

            transformation = "log1p"

        else:

            final_name = feature

            transformation = "none"

        manifest_rows.append(
            {
                "feature_name": final_name,
                "source_feature": feature,
                "transformation": transformation,
                "category": (
                    "activity"
                    if feature in [
                        "in_transaction_count",
                        "out_transaction_count",
                    ]
                    else
                    "network"
                    if feature in [
                        "unique_senders",
                        "unique_receivers",
                        "reciprocal_counterparty_count",
                        "reciprocity_ratio",
                        "counterparty_concentration",
                    ]
                    else
                    "temporal"
                    if feature in [
                        "incoming_active_days",
                        "outgoing_active_days",
                        "active_days",
                        "activity_burst_ratio",
                    ]
                    else
                    "amount"
                    if "amount" in feature
                    or "currency_abs_z" in feature
                    else
                    "diversity"
                ),
                "leakage_risk": "none",
                "target_included": False,
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    manifest.to_csv(
        MANIFEST_FILE,
        index=False,
    )

    print("\n" + "=" * 75)
    print("FINAL FEATURE MATRIX CREATED")
    print("=" * 75)

    print(
        f"\nAccounts: "
        f"{len(final_df):,}"
    )

    print(
        f"Features: "
        f"{len(final_df.columns) - 1}"
    )

    print(
        f"\nFeature matrix:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nFeature manifest:\n"
        f"{MANIFEST_FILE}"
    )

    print("\nFinal features:")

    for feature in final_df.columns:
        if feature != "account_id":
            print(f"  - {feature}")

    print("\nTarget labels remain separate:")
    print(
        "  data/processed/account_labels.csv"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()