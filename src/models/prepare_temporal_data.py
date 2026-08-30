from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "HI-Small_Trans_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "temporal_experiment.pt"
)

RANDOM_STATE = 42


# =========================================================
# Utility
# =========================================================

def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> np.ndarray:

    return np.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )


# =========================================================
# Build features for ONE time window
# =========================================================

def build_period_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy()

    # -----------------------------------------------------
    # Account IDs
    # -----------------------------------------------------

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
        {
            "account_id": accounts
        }
    )

    # -----------------------------------------------------
    # Basic activity
    # -----------------------------------------------------

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

    for column in [
        "in_transaction_count",
        "out_transaction_count",
        "unique_senders",
        "unique_receivers",
    ]:

        features[column] = (
            features[column]
            .fillna(0)
            .astype(np.int32)
        )

    # -----------------------------------------------------
    # Network
    # -----------------------------------------------------

    features["reciprocal_counterparty_count"] = 0

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
                set(),
            ).add(dst)

            reciprocal_counterparties.setdefault(
                dst,
                set(),
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
        suffixes=(
            "",
            "_new",
        ),
    )

    if "reciprocal_counterparty_count_new" in features:

        features[
            "reciprocal_counterparty_count"
        ] = features[
            "reciprocal_counterparty_count_new"
        ]

        features.drop(
            columns=[
                "reciprocal_counterparty_count_new"
            ],
            inplace=True,
        )

    features[
        "reciprocal_counterparty_count"
    ] = (
        features[
            "reciprocal_counterparty_count"
        ]
        .fillna(0)
        .astype(np.int16)
    )

    unique_counterparties = (
        features["unique_senders"]
        + features["unique_receivers"]
    )

    features["reciprocity_ratio"] = safe_ratio(
        features[
            "reciprocal_counterparty_count"
        ],
        unique_counterparties,
    )

    # -----------------------------------------------------
    # Temporal
    # -----------------------------------------------------

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

    for column in [
        "incoming_active_days",
        "outgoing_active_days",
    ]:

        features[column] = (
            features[column]
            .fillna(0)
            .astype(np.int16)
        )

    incoming_dates = (
        df[
            [
                "to_account_id",
                "date",
            ]
        ]
        .rename(
            columns={
                "to_account_id": "account_id"
            }
        )
    )

    outgoing_dates = (
        df[
            [
                "from_account_id",
                "date",
            ]
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
        .astype(np.int16)
    )

    # Daily activity.
    incoming_daily = (
        df.groupby(
            [
                "to_account_id",
                "date",
            ]
        )
        .size()
        .reset_index(
            name="daily_in"
        )
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
        .reset_index(
            name="daily_out"
        )
        .rename(
            columns={
                "from_account_id": "account_id"
            }
        )
    )

    daily = pd.merge(
        incoming_daily,
        outgoing_daily,
        on=[
            "account_id",
            "date",
        ],
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

    for column in [
        "max_daily_in_transactions",
        "max_daily_out_transactions",
        "max_daily_transactions",
    ]:

        features[column] = (
            features[column]
            .fillna(0)
            .astype(np.int32)
        )

    features["transactions_per_active_day"] = (
        safe_ratio(
            features[
                "in_transaction_count"
            ]
            + features[
                "out_transaction_count"
            ],
            features[
                "active_days"
            ],
        )
    )

    average_daily_activity = features[
        "transactions_per_active_day"
    ]

    features["activity_burst_ratio"] = safe_ratio(
        features[
            "max_daily_transactions"
        ],
        average_daily_activity,
    )

    # -----------------------------------------------------
    # Bank / payment diversity
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Counterparty concentration
    # -----------------------------------------------------

    incoming_pairs = (
        df.groupby(
            [
                "to_account_id",
                "from_account_id",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
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
        .reset_index(
            name="count"
        )
        .rename(
            columns={
                "from_account_id": "account_id"
            }
        )
    )

    pair_counts = pd.concat(
        [
            incoming_pairs[
                [
                    "account_id",
                    "count",
                ]
            ],
            outgoing_pairs[
                [
                    "account_id",
                    "count",
                ]
            ],
        ],
        ignore_index=True,
    )

    pair_counts["share"] = (
        pair_counts["count"]
        /
        pair_counts.groupby(
            "account_id"
        )["count"].transform("sum")
    )

    concentration = (
        pair_counts.assign(
            share_squared=(
                pair_counts["share"] ** 2
            )
        )
        .groupby("account_id")[
            "share_squared"
        ]
        .sum()
        .rename(
            "counterparty_concentration"
        )
    )

    features = features.merge(
        concentration,
        left_on="account_id",
        right_index=True,
        how="left",
    )

    # -----------------------------------------------------
    # Amount features
    # -----------------------------------------------------

    df["received_log"] = np.log1p(
        df["amount_received"].clip(
            lower=0
        )
    )

    df["paid_log"] = np.log1p(
        df["amount_paid"].clip(
            lower=0
        )
    )

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

    df["received_currency_z"] = (
        safe_ratio(
            df["received_log"]
            - df["received_currency_mean"],
            df["received_currency_std"]
            .fillna(0),
        )
    )

    df["paid_currency_z"] = (
        safe_ratio(
            df["paid_log"]
            - df["paid_currency_mean"],
            df["paid_currency_std"]
            .fillna(0),
        )
    )

    df["received_currency_abs_z"] = (
        df["received_currency_z"].abs()
    )

    df["paid_currency_abs_z"] = (
        df["paid_currency_z"].abs()
    )

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

    # -----------------------------------------------------
    # Feature cleanup
    # -----------------------------------------------------

    numeric_columns = [
        column
        for column in features.columns
        if column != "account_id"
    ]

    features[numeric_columns] = (
        features[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    # -----------------------------------------------------
    # Create the same final 26-feature representation
    # -----------------------------------------------------

    final_features = [
        "in_transaction_count",
        "out_transaction_count",
        "unique_senders",
        "unique_receivers",
        "reciprocal_counterparty_count",
        "reciprocity_ratio",
        "counterparty_concentration",
        "incoming_active_days",
        "outgoing_active_days",
        "active_days",
        "activity_burst_ratio",
        "unique_sender_banks",
        "unique_receiver_banks",
        "incoming_payment_format_diversity",
        "outgoing_payment_format_diversity",
        "receiving_currency_diversity",
        "payment_currency_diversity",
        "incoming_log_amount_mean",
        "incoming_log_amount_std",
        "incoming_log_amount_max",
        "incoming_currency_abs_z_mean",
        "incoming_currency_abs_z_max",
        "outgoing_log_amount_mean",
        "outgoing_log_amount_std",
        "outgoing_log_amount_max",
        "outgoing_currency_abs_z_max",
    ]

    missing = [
        feature
        for feature in final_features
        if feature not in features.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing period features: {missing}"
        )

    feature_table = features[
        ["account_id"] + final_features
    ].copy()

    # -----------------------------------------------------
    # Period labels
    # -----------------------------------------------------

    illicit_accounts = set()

    laundering = df[
        df["is_laundering"] == 1
    ]

    illicit_accounts.update(
        laundering["from_account_id"]
    )

    illicit_accounts.update(
        laundering["to_account_id"]
    )

    labels = pd.DataFrame(
        {
            "account_id": feature_table[
                "account_id"
            ],
            "is_illicit": (
                feature_table[
                    "account_id"
                ].isin(illicit_accounts)
                .astype(np.int64)
            ),
        }
    )

    return (
        feature_table,
        labels,
    )


# =========================================================
# Build graph from one period
# =========================================================

def build_graph(
    df: pd.DataFrame,
    feature_table: pd.DataFrame,
    labels: pd.DataFrame,
) -> Data:

    # -----------------------------------------------------
    # Create globally unique account IDs for this period
    # -----------------------------------------------------

    df = df.copy()

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

    account_ids = feature_table[
        "account_id"
    ].tolist()

    node_mapping = {
        account_id: index
        for index, account_id
        in enumerate(account_ids)
    }

    src = df[
        "from_account_id"
    ].map(node_mapping)

    dst = df[
        "to_account_id"
    ].map(node_mapping)

    valid = (
        src.notna()
        & dst.notna()
    )

    src_array = (
        src.loc[valid]
        .to_numpy(
            dtype=np.int64
        )
    )

    dst_array = (
        dst.loc[valid]
        .to_numpy(
            dtype=np.int64
        )
    )

    edge_index = torch.from_numpy(
        np.vstack(
            [
                src_array,
                dst_array,
            ]
        ).copy()
    )

    label_array = labels[
        "is_illicit"
    ].to_numpy(
        dtype=np.int64
    )

    return (
        Data(
            edge_index=edge_index,
            y=torch.from_numpy(
                label_array.copy()
            ),
            num_nodes=len(account_ids),
        ),
        account_ids,
    )


# =========================================================
# Main
# =========================================================

def main():

    start = time.perf_counter()

    print("=" * 75)
    print("PHASE 7F — CHRONOLOGICAL DATA PREPARATION")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    print("\n[1/7] Loading cleaned transactions...")

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
        "is_laundering",
    ]

    df = pd.read_csv(
        INPUT_FILE,
        usecols=columns,
        parse_dates=[
            "timestamp"
        ],
    )

    print(
        f"Transactions: {len(df):,}"
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    print(
        "\n[2/7] Sorting transactions by timestamp..."
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Determine boundaries
    # -----------------------------------------------------

    print(
        "\n[3/7] Determining chronological boundaries..."
    )

    train_cutoff = df[
        "timestamp"
    ].quantile(
        0.70
    )

    test_cutoff = df[
        "timestamp"
    ].quantile(
        0.85
    )

    train_df = df[
        df["timestamp"]
        < train_cutoff
    ].copy()

    val_df = df[
        (
            df["timestamp"]
            >= train_cutoff
        )
        &
        (
            df["timestamp"]
            < test_cutoff
        )
    ].copy()

    test_df = df[
        df["timestamp"]
        >= test_cutoff
    ].copy()

    print(
        f"Train cutoff: {train_cutoff}"
    )

    print(
        f"Test cutoff:  {test_cutoff}"
    )

    print(
        f"\nTrain transactions: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation transactions: "
        f"{len(val_df):,}"
    )

    print(
        f"Test transactions: "
        f"{len(test_df):,}"
    )

    # -----------------------------------------------------
    # Build period features
    # -----------------------------------------------------

    print(
        "\n[4/7] Building period-specific features..."
    )

    train_features, train_labels = (
        build_period_features(
            train_df
        )
    )

    val_features, val_labels = (
        build_period_features(
            val_df
        )
    )

    test_features, test_labels = (
        build_period_features(
            test_df
        )
    )

    print(
        f"Train accounts: "
        f"{len(train_features):,}"
    )

    print(
        f"Validation accounts: "
        f"{len(val_features):,}"
    )

    print(
        f"Test accounts: "
        f"{len(test_features):,}"
    )

    # -----------------------------------------------------
    # Fit scaler on TRAIN ONLY
    # -----------------------------------------------------

    print(
        "\n[5/7] Fitting scaler on TRAIN only..."
    )

    feature_columns = [
        column
        for column in train_features.columns
        if column != "account_id"
    ]

    scaler = StandardScaler()

    train_matrix = scaler.fit_transform(
        train_features[
            feature_columns
        ]
    ).astype(
        np.float32
    )

    val_matrix = scaler.transform(
        val_features[
            feature_columns
        ]
    ).astype(
        np.float32
    )

    test_matrix = scaler.transform(
        test_features[
            feature_columns
        ]
    ).astype(
        np.float32
    )

    # -----------------------------------------------------
    # Build graph structures
    # -----------------------------------------------------

    print(
        "\n[6/7] Building temporal graph objects..."
    )

    train_graph, train_accounts = build_graph(
        train_df,
        train_features,
        train_labels,
    )

    val_graph, val_accounts = build_graph(
        val_df,
        val_features,
        val_labels,
    )

    test_graph, test_accounts = build_graph(
        test_df,
        test_features,
        test_labels,
    )

    train_graph.x = torch.from_numpy(
        train_matrix.copy()
    )

    val_graph.x = torch.from_numpy(
        val_matrix.copy()
    )

    test_graph.x = torch.from_numpy(
        test_matrix.copy()
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print(
        "\n[7/7] Saving temporal experiment..."
    )

    torch.save(
        {
            "train": train_graph,
            "validation": val_graph,
            "test": test_graph,

            "train_accounts": train_accounts,
            "validation_accounts": val_accounts,
            "test_accounts": test_accounts,

            "feature_columns": feature_columns,

            "train_cutoff": str(
                train_cutoff
            ),

            "test_cutoff": str(
                test_cutoff
            ),

            "scaler": scaler,

            "description": (
                "Strict chronological "
                "70/15/15 transaction split. "
                "Features and graphs built "
                "separately per period. "
                "Scaler fitted on train only."
            ),
        },
        OUTPUT_FILE,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start
    )

    print("\n" + "=" * 75)
    print("PHASE 7F — TEMPORAL PREPARATION COMPLETE")
    print("=" * 75)

    print(
        f"\nTRAIN"
    )

    print(
        f"  Transactions: "
        f"{len(train_df):,}"
    )

    print(
        f"  Accounts:     "
        f"{len(train_graph.y):,}"
    )

    print(
        f"  Edges:        "
        f"{train_graph.edge_index.shape[1]:,}"
    )

    print(
        f"  Illicit:      "
        f"{int(train_graph.y.sum()):,}"
    )

    print(
        f"\nVALIDATION"
    )

    print(
        f"  Transactions: "
        f"{len(val_df):,}"
    )

    print(
        f"  Accounts:     "
        f"{len(val_graph.y):,}"
    )

    print(
        f"  Edges:        "
        f"{val_graph.edge_index.shape[1]:,}"
    )

    print(
        f"  Illicit:      "
        f"{int(val_graph.y.sum()):,}"
    )

    print(
        f"\nTEST"
    )

    print(
        f"  Transactions: "
        f"{len(test_df):,}"
    )

    print(
        f"  Accounts:     "
        f"{len(test_graph.y):,}"
    )

    print(
        f"  Edges:        "
        f"{test_graph.edge_index.shape[1]:,}"
    )

    print(
        f"  Illicit:      "
        f"{int(test_graph.y.sum()):,}"
    )

    print(
        f"\nFeatures: "
        f"{len(feature_columns)}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print(
        f"Runtime: {elapsed:.2f} seconds"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()