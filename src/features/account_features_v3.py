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
    / "account_features_v3.csv"
)


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return np.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )


def build_features() -> None:

    start = time.perf_counter()

    print("=" * 75)
    print("PHASE 4 — ACCOUNT FEATURE ENGINEERING v3")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load transactions
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
    # 2. Create account IDs
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

    print("\n[3/8] Basic and topology features...")

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

    for series in [
        in_count,
        out_count,
        unique_senders,
        unique_receivers,
    ]:
        features = features.merge(
            series,
            left_on="account_id",
            right_index=True,
            how="left",
        )

    for col in [
        "in_transaction_count",
        "out_transaction_count",
        "unique_senders",
        "unique_receivers",
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

    features["unique_counterparties"] = (
        features["unique_senders"]
        + features["unique_receivers"]
    )

    features["fan_in"] = features["unique_senders"]
    features["fan_out"] = features["unique_receivers"]

    # These replace the duplicate terminology from v2.
    features["fan_in_intensity"] = safe_ratio(
        features["fan_in"],
        features["in_transaction_count"],
    )

    features["fan_out_intensity"] = safe_ratio(
        features["fan_out"],
        features["out_transaction_count"],
    )

    features["in_out_transaction_ratio"] = safe_ratio(
        features["in_transaction_count"],
        features["out_transaction_count"],
    )

    # ---------------------------------------------------------
    # 4. Temporal features
    # ---------------------------------------------------------

    print("\n[4/8] Temporal features...")

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

    for series in [
        incoming_days,
        outgoing_days,
    ]:
        features = features.merge(
            series,
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

    incoming_dates = (
        df[
            ["to_account_id", "date"]
        ]
        .rename(
            columns={
                "to_account_id": "account_id"
            }
        )
    )

    outgoing_dates = (
        df[
            ["from_account_id", "date"]
        ]
        .rename(
            columns={
                "from_account_id": "account_id"
            }
        )
    )

    active_dates = (
        pd.concat(
            [
                incoming_dates,
                outgoing_dates,
            ],
            ignore_index=True,
        )
        .drop_duplicates()
    )

    active_days = (
        active_dates
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

    incoming_daily = (
        df.groupby(
            [
                "to_account_id",
                "date",
            ]
        )
        .size()
        .rename("daily_in")
        .reset_index()
        .rename(
            columns={
                "to_account_id": "account_id"
            }
        )
    )

    outgoing_daily = (
        df.groupby(
            [
                "from_account_id",
                "date",
            ]
        )
        .size()
        .rename("daily_out")
        .reset_index()
        .rename(
            columns={
                "from_account_id": "account_id"
            }
        )
    )

    daily = pd.merge(
        incoming_daily,
        outgoing_daily,
        on=["account_id", "date"],
        how="outer",
    ).fillna(0)

    daily["daily_total"] = (
        daily["daily_in"]
        + daily["daily_out"]
    )

    max_daily = (
        daily.groupby("account_id")[
            [
                "daily_in",
                "daily_out",
                "daily_total",
            ]
        ]
        .max()
        .rename(
            columns={
                "daily_in":
                    "max_daily_in_transactions",
                "daily_out":
                    "max_daily_out_transactions",
                "daily_total":
                    "max_daily_transactions",
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

    features["activity_burst_ratio"] = safe_ratio(
        features["max_daily_transactions"],
        features["transactions_per_active_day"],
    )

    # ---------------------------------------------------------
    # 5. Bank / payment diversity
    # ---------------------------------------------------------

    print("\n[5/8] Bank and payment diversity...")

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

    incoming_format = (
        df.groupby("to_account_id")["payment_format"]
        .nunique()
        .rename(
            "incoming_payment_format_diversity"
        )
    )

    outgoing_format = (
        df.groupby("from_account_id")["payment_format"]
        .nunique()
        .rename(
            "outgoing_payment_format_diversity"
        )
    )

    receiving_currency = (
        df.groupby("to_account_id")[
            "receiving_currency"
        ]
        .nunique()
        .rename(
            "receiving_currency_diversity"
        )
    )

    payment_currency = (
        df.groupby("from_account_id")[
            "payment_currency"
        ]
        .nunique()
        .rename(
            "payment_currency_diversity"
        )
    )

    for series in [
        sender_banks,
        receiver_banks,
        incoming_format,
        outgoing_format,
        receiving_currency,
        payment_currency,
    ]:
        features = features.merge(
            series,
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

    # ---------------------------------------------------------
    # 6. Reciprocal behavior
    # ---------------------------------------------------------

    print("\n[6/8] Reciprocal relationships...")

    directed_pairs = (
        df[
            [
                "from_account_id",
                "to_account_id",
            ]
        ]
        .drop_duplicates()
    )

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

    reciprocal_counterparties = {}

    for src, dst in pair_set:

        if (dst, src) in pair_set:

            reciprocal_counterparties.setdefault(
                src,
                set()
            ).add(dst)

            reciprocal_counterparties.setdefault(
                dst,
                set()
            ).add(src)

    reciprocal_counts = pd.Series(
        {
            account: len(counterparties)
            for account, counterparties
            in reciprocal_counterparties.items()
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
        .astype("int16")
    )

    features["reciprocity_ratio"] = safe_ratio(
        features["reciprocal_counterparty_count"],
        features["unique_counterparties"],
    )

    # ---------------------------------------------------------
    # Counterparty concentration
    # ---------------------------------------------------------

    print("\nCalculating counterparty concentration...")

    incoming_pairs = (
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

    outgoing_pairs = (
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
            incoming_pairs[
                ["account_id", "count"]
            ],
            outgoing_pairs[
                ["account_id", "count"]
            ],
        ],
        ignore_index=True,
    )

    pair_counts["share"] = (
        pair_counts["count"]
        /
        pair_counts.groupby("account_id")["count"]
        .transform("sum")
    )

    counterparty_concentration = (
        pair_counts.assign(
            share_squared=pair_counts["share"] ** 2
        )
        .groupby("account_id")["share_squared"]
        .sum()
        .rename("counterparty_concentration")
    )

    features = features.merge(
        counterparty_concentration,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    # ---------------------------------------------------------
    # 7. Amount behavior
    # ---------------------------------------------------------

    print("\n[7/8] Currency-aware amount features...")

    # Log transform first because financial amounts are usually
    # heavily right-skewed.
    df["received_log"] = np.log1p(
        df["amount_received"].clip(lower=0)
    )

    df["paid_log"] = np.log1p(
        df["amount_paid"].clip(lower=0)
    )

    # Currency-relative normalization.
    #
    # IMPORTANT:
    # These statistics are calculated over the entire dataset
    # for the exploratory Phase-4 matrix.
    #
    # Before model training we must recompute them using TRAIN
    # data only to avoid temporal leakage.
    received_stats = (
        df.groupby("receiving_currency")[
            "received_log"
        ]
        .agg(
            received_currency_mean="mean",
            received_currency_std="std",
        )
    )

    paid_stats = (
        df.groupby("payment_currency")[
            "paid_log"
        ]
        .agg(
            paid_currency_mean="mean",
            paid_currency_std="std",
        )
    )

    df = df.join(
        received_stats,
        on="receiving_currency",
    )

    df = df.join(
        paid_stats,
        on="payment_currency",
    )

    df["received_currency_z"] = safe_ratio(
        df["received_log"]
        - df["received_currency_mean"],
        df["received_currency_std"],
    )

    df["paid_currency_z"] = safe_ratio(
        df["paid_log"]
        - df["paid_currency_mean"],
        df["paid_currency_std"],
    )

    # Absolute deviation is often more useful than signed z-score
    # for anomaly detection.
    df["received_currency_abs_z"] = (
        df["received_currency_z"]
        .abs()
    )

    df["paid_currency_abs_z"] = (
        df["paid_currency_z"]
        .abs()
    )

    # Incoming amount aggregates.
    incoming_amounts = (
        df.groupby("to_account_id")
        .agg(
            incoming_log_amount_mean=(
                "received_log",
                "mean",
            ),
            incoming_log_amount_std=(
                "received_log",
                "std",
            ),
            incoming_log_amount_max=(
                "received_log",
                "max",
            ),
            incoming_currency_abs_z_mean=(
                "received_currency_abs_z",
                "mean",
            ),
            incoming_currency_abs_z_max=(
                "received_currency_abs_z",
                "max",
            ),
        )
    )

    # Outgoing amount aggregates.
    outgoing_amounts = (
        df.groupby("from_account_id")
        .agg(
            outgoing_log_amount_mean=(
                "paid_log",
                "mean",
            ),
            outgoing_log_amount_std=(
                "paid_log",
                "std",
            ),
            outgoing_log_amount_max=(
                "paid_log",
                "max",
            ),
            outgoing_currency_abs_z_mean=(
                "paid_currency_abs_z",
                "mean",
            ),
            outgoing_currency_abs_z_max=(
                "paid_currency_abs_z",
                "max",
            ),
        )
    )

    features = features.merge(
        incoming_amounts,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    features = features.merge(
        outgoing_amounts,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    # ---------------------------------------------------------
    # 8. Cleanup and save
    # ---------------------------------------------------------

    print("\n[8/8] Finalizing feature matrix...")

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
    print("PHASE 4 — FEATURE ENGINEERING v3 COMPLETE")
    print("=" * 75)

    print(
        f"\nAccounts: {len(features):,}"
    )

    print(
        f"Features: {len(features.columns) - 1}"
    )

    print(
        f"Runtime:  {elapsed:.2f} seconds"
    )

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )

    print("\nGenerated features:")

    for col in features.columns:
        if col != "account_id":
            print(f"  - {col}")

    print("=" * 75)


if __name__ == "__main__":
    build_features()