from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "HI-Small_Trans_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_features_v2.csv"
)


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Element-wise ratio with zero-safe denominator handling."""
    return np.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )


def build_features() -> None:

    start = time.perf_counter()

    print("=" * 75)
    print("PHASE 4 — ACCOUNT FEATURE ENGINEERING v2")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load
    # ---------------------------------------------------------

    print("\n[1/8] Loading transactions...")

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

    print(f"Transactions: {len(df):,}")

    # ---------------------------------------------------------
    # 2. Account IDs
    # ---------------------------------------------------------

    print("\n[2/8] Creating account IDs...")

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

    accounts = pd.unique(
        pd.concat(
            [
                df["from_account_id"],
                df["to_account_id"],
            ],
            ignore_index=True,
        )
    )

    features = pd.DataFrame(
        {"account_id": accounts}
    )

    print(f"Accounts: {len(features):,}")

    # ---------------------------------------------------------
    # 3. Basic activity
    # ---------------------------------------------------------

    print("\n[3/8] Basic activity features...")

    in_count = (
        df.groupby("to_account_id")
        .size()
        .rename("in_transaction_count")
    )

    out_count = (
        df.groupby("from_account_id")
        .size()
        .rename("out_transaction_count")
    )

    features = features.merge(
        in_count,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        out_count,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    for col in [
        "in_transaction_count",
        "out_transaction_count",
    ]:
        features[col] = (
            features[col]
            .fillna(0)
            .astype("int32")
        )

    features["total_transaction_count"] = (
        features["in_transaction_count"]
        + features["out_transaction_count"]
    )

    features["in_out_transaction_ratio"] = safe_ratio(
        features["in_transaction_count"],
        features["out_transaction_count"],
    )

    # ---------------------------------------------------------
    # 4. Counterparty / topology
    # ---------------------------------------------------------

    print("\n[4/8] Counterparty and topology features...")

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

    for col in [
        "unique_senders",
        "unique_receivers",
    ]:
        features[col] = (
            features[col]
            .fillna(0)
            .astype("int32")
        )

    features["unique_counterparties"] = (
        features["unique_senders"]
        + features["unique_receivers"]
    )

    # Explicit semantic names.
    features["fan_in"] = features["unique_senders"]
    features["fan_out"] = features["unique_receivers"]

    features["transaction_in_degree"] = (
        features["in_transaction_count"]
    )

    features["transaction_out_degree"] = (
        features["out_transaction_count"]
    )

    features["in_out_counterparty_ratio"] = safe_ratio(
        features["unique_senders"],
        features["unique_receivers"],
    )

    # ---------------------------------------------------------
    # Reciprocal relationships — corrected implementation
    # ---------------------------------------------------------

    print("  Calculating reciprocal relationships...")

    directed_pairs = (
        df[
            [
                "from_account_id",
                "to_account_id",
            ]
        ]
        .drop_duplicates()
    )

    # Remove self relationships.
    directed_pairs = directed_pairs[
        directed_pairs["from_account_id"]
        != directed_pairs["to_account_id"]
    ]

    pair_set = set(
        zip(
            directed_pairs["from_account_id"],
            directed_pairs["to_account_id"],
        )
    )

    reciprocal_accounts = {}

    for src, dst in pair_set:
        if (dst, src) in pair_set:
            reciprocal_accounts.setdefault(src, set()).add(dst)
            reciprocal_accounts.setdefault(dst, set()).add(src)

    reciprocal_counts = pd.Series(
        {
            account: len(counterparties)
            for account, counterparties
            in reciprocal_accounts.items()
        },
        name="reciprocal_counterparty_count",
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

    features["reciprocity_ratio"] = safe_ratio(
        features["reciprocal_counterparty_count"],
        features["unique_counterparties"],
    )

    # ---------------------------------------------------------
    # 5. Temporal features
    # ---------------------------------------------------------

    print("\n[5/8] Temporal behavior features...")

    df["date"] = df["timestamp"].dt.date

    # Active days.
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

    for col in [
        "incoming_active_days",
        "outgoing_active_days",
    ]:
        features[col] = (
            features[col]
            .fillna(0)
            .astype("int16")
        )

    # True active days = union of incoming/outgoing dates.
    incoming_dates = (
        df[["to_account_id", "date"]]
        .rename(columns={"to_account_id": "account_id"})
    )

    outgoing_dates = (
        df[["from_account_id", "date"]]
        .rename(columns={"from_account_id": "account_id"})
    )

    all_account_dates = pd.concat(
        [
            incoming_dates,
            outgoing_dates,
        ],
        ignore_index=True,
    ).drop_duplicates()

    active_days = (
        all_account_dates
        .groupby("account_id")
        .size()
        .rename("active_days")
    )

    features = features.merge(
        active_days,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features["active_days"] = (
        features["active_days"]
        .fillna(0)
        .astype("int16")
    )

    features["transactions_per_active_day"] = safe_ratio(
        features["total_transaction_count"],
        features["active_days"],
    )

    # ---------------------------------------------------------
    # Daily transaction counts
    # ---------------------------------------------------------

    incoming_daily = (
        df.groupby(
            [
                "to_account_id",
                "date",
            ]
        )
        .size()
        .rename("daily_in_count")
        .reset_index()
        .rename(columns={"to_account_id": "account_id"})
    )

    outgoing_daily = (
        df.groupby(
            [
                "from_account_id",
                "date",
            ]
        )
        .size()
        .rename("daily_out_count")
        .reset_index()
        .rename(columns={"from_account_id": "account_id"})
    )

    daily = pd.merge(
        incoming_daily,
        outgoing_daily,
        on=["account_id", "date"],
        how="outer",
    ).fillna(0)

    daily["daily_total_count"] = (
        daily["daily_in_count"]
        + daily["daily_out_count"]
    )

    max_daily = (
        daily.groupby("account_id")[
            [
                "daily_in_count",
                "daily_out_count",
                "daily_total_count",
            ]
        ]
        .max()
        .rename(
            columns={
                "daily_in_count": "max_daily_in_transactions",
                "daily_out_count": "max_daily_out_transactions",
                "daily_total_count": "max_daily_transactions",
            }
        )
    )

    features = features.merge(
        max_daily,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    for col in [
        "max_daily_in_transactions",
        "max_daily_out_transactions",
        "max_daily_transactions",
    ]:
        features[col] = (
            features[col]
            .fillna(0)
            .astype("int32")
        )

    # Burst ratio:
    # max daily activity / average daily activity.
    features["activity_burst_ratio"] = safe_ratio(
        features["max_daily_transactions"],
        features["transactions_per_active_day"],
    )

    # ---------------------------------------------------------
    # 6. Cross-bank behavior
    # ---------------------------------------------------------

    print("\n[6/8] Cross-bank and diversity features...")

    sender_banks = (
        df.groupby("to_account_id")["from_bank"]
        .nunique()
        .rename("unique_sender_banks")
    )

    receiver_banks = (
        df.groupby("from_account_id")["to_bank"]
        .nunique()
        .rename("unique_receiver_banks")
    )

    features = features.merge(
        sender_banks,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        receiver_banks,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    for col in [
        "unique_sender_banks",
        "unique_receiver_banks",
    ]:
        features[col] = (
            features[col]
            .fillna(0)
            .astype("int16")
        )

    features["total_counterparty_banks"] = (
        features["unique_sender_banks"]
        + features["unique_receiver_banks"]
    )

    # ---------------------------------------------------------
    # Payment format / currency diversity
    # ---------------------------------------------------------

    payment_in = (
        df.groupby("to_account_id")["payment_format"]
        .nunique()
        .rename("incoming_payment_format_diversity")
    )

    payment_out = (
        df.groupby("from_account_id")["payment_format"]
        .nunique()
        .rename("outgoing_payment_format_diversity")
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

    for series in [
        payment_in,
        payment_out,
        currency_in,
        currency_out,
    ]:
        features = features.merge(
            series,
            left_on="account_id",
            right_index=True,
            how="left",
        )

    for col in [
        "incoming_payment_format_diversity",
        "outgoing_payment_format_diversity",
        "receiving_currency_diversity",
        "payment_currency_diversity",
    ]:
        features[col] = (
            features[col]
            .fillna(0)
            .astype("int8")
        )

    # Cross-bank ratio.
    features["cross_bank_ratio"] = safe_ratio(
        features["total_counterparty_banks"],
        features["unique_counterparties"],
    )

    # ---------------------------------------------------------
    # 7. Counterparty concentration
    # ---------------------------------------------------------

    print("\n[7/8] Counterparty concentration...")

    # Number of transactions exchanged with each counterparty.
    incoming_pair_counts = (
        df.groupby(
            [
                "to_account_id",
                "from_account_id",
            ]
        )
        .size()
        .reset_index(name="count")
        .rename(
            columns={
                "to_account_id": "account_id"
            }
        )
    )

    outgoing_pair_counts = (
        df.groupby(
            [
                "from_account_id",
                "to_account_id",
            ]
        )
        .size()
        .reset_index(name="count")
        .rename(
            columns={
                "from_account_id": "account_id"
            }
        )
    )

    pair_counts = pd.concat(
        [
            incoming_pair_counts[
                ["account_id", "count"]
            ],
            outgoing_pair_counts[
                ["account_id", "count"]
            ],
        ],
        ignore_index=True,
    )

    # For each account calculate the share of activity represented
    # by each counterparty. HHI = sum(p_i^2).
    pair_counts["share"] = (
        pair_counts["count"]
        /
        pair_counts.groupby("account_id")["count"]
        .transform("sum")
    )

    concentration = (
        pair_counts.assign(
            share_squared=pair_counts["share"] ** 2
        )
        .groupby("account_id")["share_squared"]
        .sum()
        .rename("counterparty_concentration")
    )

    features = features.merge(
        concentration,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    # ---------------------------------------------------------
    # 8. Cleanup + save
    # ---------------------------------------------------------

    print("\n[8/8] Cleaning feature table and saving...")

    numeric_columns = [
        col
        for col in features.columns
        if col != "account_id"
    ]

    features[numeric_columns] = (
        features[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

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
    print("PHASE 4 — FEATURE ENGINEERING v2 COMPLETE")
    print("=" * 75)

    print(f"\nAccounts:  {len(features):,}")
    print(f"Features:  {len(features.columns) - 1:,}")
    print(f"Runtime:   {elapsed:.2f} seconds")

    print(f"\nOutput:")
    print(OUTPUT_FILE)

    print("\nFeatures:")
    for col in features.columns:
        if col != "account_id":
            print(f"  - {col}")

    print("=" * 75)


if __name__ == "__main__":
    build_features()