from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd


print("=" * 75)
print("PHASE 4 — ACCOUNT FEATURE ENGINEERING")
print("=" * 75)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "HI-Small_Trans_clean.csv"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_labels.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_features_full.csv"
)


def build_account_features() -> None:

    start = time.perf_counter()


    print("\nLoading transactions...")

    columns = [
        "timestamp",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
    ]

    df = pd.read_csv(
        INPUT_FILE,
        usecols=columns,
        parse_dates=["timestamp"],
    )

    print(
        f"Transactions loaded: {len(df):,}"
    )

    # -----------------------------------------------------
    # Create globally unique account IDs
    # -----------------------------------------------------

    print("\nCreating account IDs...")

    df["from_account_id"] = (
        df["from_bank"].astype(str)
        + "_"
        + df["from_account"].astype(str)
    )

    df["to_account_id"] = (
        df["to_bank"].astype(str)
        + "_"
        + df["to_account"].astype(str)
    )

    # -----------------------------------------------------
    # Build complete account list
    # -----------------------------------------------------

    accounts = pd.unique(
        pd.concat(
            [
                df["from_account_id"],
                df["to_account_id"],
            ],
            ignore_index=True,
        )
    )

    features = pd.DataFrame({
        "account_id": accounts
    })

    print(
        f"Accounts: {len(features):,}"
    )

    # -----------------------------------------------------
    # BASIC TRANSACTION COUNTS
    # -----------------------------------------------------

    print("\n[1/7] Basic transaction counts...")

    incoming_count = (
        df.groupby("to_account_id")
        .size()
        .rename("in_transaction_count")
    )

    outgoing_count = (
        df.groupby("from_account_id")
        .size()
        .rename("out_transaction_count")
    )

    features = features.merge(
        incoming_count,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        outgoing_count,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features["in_transaction_count"] = (
        features["in_transaction_count"]
        .fillna(0)
        .astype("int32")
    )

    features["out_transaction_count"] = (
        features["out_transaction_count"]
        .fillna(0)
        .astype("int32")
    )

    features["total_transaction_count"] = (
        features["in_transaction_count"]
        + features["out_transaction_count"]
    )

    # -----------------------------------------------------
    # UNIQUE COUNTERPARTIES
    # -----------------------------------------------------

    print("[2/7] Counterparty features...")

    unique_senders = (
        df.groupby("to_account_id")["from_account_id"]
        .nunique()
        .rename("unique_senders")
    )

    unique_receivers = (
        df.groupby("from_account_id")["to_account_id"]
        .nunique()
        .rename("unique_receivers")
    )

    features = features.merge(
        unique_senders,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        unique_receivers,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features["unique_senders"] = (
        features["unique_senders"]
        .fillna(0)
        .astype("int32")
    )

    features["unique_receivers"] = (
        features["unique_receivers"]
        .fillna(0)
        .astype("int32")
    )

    features["unique_counterparties"] = (
        features["unique_senders"]
        + features["unique_receivers"]
    )

    # -----------------------------------------------------
    # DEGREE / FAN-IN / FAN-OUT
    # -----------------------------------------------------

    print("[3/7] Network topology features...")

    # In this account graph:
    # fan-in  = number of distinct senders
    # fan-out = number of distinct receivers

    features["fan_in"] = features["unique_senders"]
    features["fan_out"] = features["unique_receivers"]

    features["in_degree"] = features["in_transaction_count"]
    features["out_degree"] = features["out_transaction_count"]

    # -----------------------------------------------------
    # RECIPROCITY
    #
    # Count distinct counterparties where both directions
    # exist: A -> B AND B -> A.
    # -----------------------------------------------------

    print("[4/7] Reciprocal relationship features...")

    directed_pairs = df[
        [
            "from_account_id",
            "to_account_id",
        ]
    ].drop_duplicates()

    reverse_pairs = directed_pairs.rename(
        columns={
            "from_account_id": "to_account_id",
            "to_account_id": "from_account_id",
        }
    )

    reciprocal_pairs = directed_pairs.merge(
        reverse_pairs,
        on=[
            "from_account_id",
            "to_account_id",
        ],
        how="inner",
    )

    # Remove self-pairs.
    reciprocal_pairs = reciprocal_pairs[
        reciprocal_pairs["from_account_id"]
        != reciprocal_pairs["to_account_id"]
    ]

    reciprocal_counts = pd.concat(
        [
            reciprocal_pairs["from_account_id"],
            reciprocal_pairs["to_account_id"],
        ],
        ignore_index=True,
    ).value_counts()

    reciprocal_counts = reciprocal_counts.rename(
        "reciprocal_counterparty_count"
    )

    features = features.merge(
        reciprocal_counts,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features["reciprocal_counterparty_count"] = (
        features["reciprocal_counterparty_count"]
        .fillna(0)
        .astype("int32")
    )

    features["reciprocity_ratio"] = np.where(
        features["unique_counterparties"] > 0,
        features["reciprocal_counterparty_count"]
        / features["unique_counterparties"],
        0.0,
    )

    # -----------------------------------------------------
    # TEMPORAL FEATURES
    # -----------------------------------------------------

    print("[5/7] Temporal behavior features...")

    df["date"] = df["timestamp"].dt.date

    incoming_days = (
        df.groupby("to_account_id")["date"]
        .nunique()
        .rename("incoming_active_days")
    )

    outgoing_days = (
        df.groupby("from_account_id")["date"]
        .nunique()
        .rename("outgoing_active_days")
    )

    features = features.merge(
        incoming_days,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        outgoing_days,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features["incoming_active_days"] = (
        features["incoming_active_days"]
        .fillna(0)
        .astype("int16")
    )

    features["outgoing_active_days"] = (
        features["outgoing_active_days"]
        .fillna(0)
        .astype("int16")
    )

    features["active_days"] = (
        features["incoming_active_days"]
        .combine(
            features["outgoing_active_days"],
            max,
        )
    )

    # Transactions per active day.
    features["transactions_per_active_day"] = np.where(
        features["active_days"] > 0,
        features["total_transaction_count"]
        / features["active_days"],
        0.0,
    )

    # -----------------------------------------------------
    # MAX DAILY ACTIVITY
    # -----------------------------------------------------

    daily_in = (
        df.groupby(
            [
                "to_account_id",
                "date",
            ]
        )
        .size()
        .groupby(level=0)
        .max()
        .rename("max_daily_in_transactions")
    )

    daily_out = (
        df.groupby(
            [
                "from_account_id",
                "date",
            ]
        )
        .size()
        .groupby(level=0)
        .max()
        .rename("max_daily_out_transactions")
    )

    features = features.merge(
        daily_in,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        daily_out,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features["max_daily_in_transactions"] = (
        features["max_daily_in_transactions"]
        .fillna(0)
        .astype("int32")
    )

    features["max_daily_out_transactions"] = (
        features["max_daily_out_transactions"]
        .fillna(0)
        .astype("int32")
    )

    features["max_daily_transactions"] = (
        features["max_daily_in_transactions"]
        + features["max_daily_out_transactions"]
    )

    # -----------------------------------------------------
    # BANK / PAYMENT DIVERSITY
    # -----------------------------------------------------

    print("[6/7] Bank and payment diversity...")

    incoming_banks = (
        df.groupby("to_account_id")["from_bank"]
        .nunique()
        .rename("unique_sender_banks")
    )

    outgoing_banks = (
        df.groupby("from_account_id")["to_bank"]
        .nunique()
        .rename("unique_receiver_banks")
    )

    features = features.merge(
        incoming_banks,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        outgoing_banks,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features["unique_sender_banks"] = (
        features["unique_sender_banks"]
        .fillna(0)
        .astype("int16")
    )

    features["unique_receiver_banks"] = (
        features["unique_receiver_banks"]
        .fillna(0)
        .astype("int16")
    )

    payment_formats_in = (
        df.groupby("to_account_id")["payment_format"]
        .nunique()
        .rename("incoming_payment_format_diversity")
    )

    payment_formats_out = (
        df.groupby("from_account_id")["payment_format"]
        .nunique()
        .rename("outgoing_payment_format_diversity")
    )

    features = features.merge(
        payment_formats_in,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        payment_formats_out,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features["incoming_payment_format_diversity"] = (
        features["incoming_payment_format_diversity"]
        .fillna(0)
        .astype("int8")
    )

    features["outgoing_payment_format_diversity"] = (
        features["outgoing_payment_format_diversity"]
        .fillna(0)
        .astype("int8")
    )

    currency_in = (
        df.groupby("to_account_id")["receiving_currency"]
        .nunique()
        .rename("receiving_currency_diversity")
    )

    currency_out = (
        df.groupby("from_account_id")["payment_currency"]
        .nunique()
        .rename("payment_currency_diversity")
    )

    features = features.merge(
        currency_in,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        currency_out,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features["receiving_currency_diversity"] = (
        features["receiving_currency_diversity"]
        .fillna(0)
        .astype("int8")
    )

    features["payment_currency_diversity"] = (
        features["payment_currency_diversity"]
        .fillna(0)
        .astype("int8")
    )

    # -----------------------------------------------------
    # FEATURE CLEANUP
    # -----------------------------------------------------

    print("[7/7] Finalizing feature table...")

    numeric_columns = [
        column
        for column in features.columns
        if column != "account_id"
    ]

    features[numeric_columns] = (
        features[numeric_columns]
        .fillna(0)
        .replace(
            [np.inf, -np.inf],
            0,
        )
    )

    # -----------------------------------------------------
    # DO NOT MERGE LABELS INTO MODEL FEATURES YET.
    #
    # This file is intentionally feature-only.
    # Labels remain separate.
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 75)
    print("PHASE 4 — INITIAL FEATURE ENGINEERING COMPLETE")
    print("=" * 75)

    print(
        f"\nAccounts:       {len(features):,}"
    )

    print(
        f"Features:       {len(features.columns) - 1}"
    )

    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    print(
        f"\nRuntime:        {elapsed:.2f} seconds"
    )

    print("\nFeatures generated:")

    for column in features.columns:
        if column != "account_id":
            print(f"  - {column}")

    print("=" * 75)


if __name__ == "__main__":
    build_account_features()