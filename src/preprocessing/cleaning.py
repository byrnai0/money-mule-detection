from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd


REQUIRED_COLUMNS = [
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
    "from_account_id",
    "to_account_id",
]


def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate the loaded IBM AML dataset.

    IMPORTANT:
    This function does NOT modify or delete anything.
    It only reports data-quality issues.
    """

    report = {}

    print("\n[1/12] Checking schema...")
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]
    report["missing_required_columns"] = missing_columns

    print("[2/12] Checking dataframe shape...")
    report["rows"] = len(df)
    report["columns"] = len(df.columns)

    print("[3/12] Checking missing values...")
    missing_values = df[REQUIRED_COLUMNS].isna().sum()
    report["missing_values"] = {
        col: int(count)
        for col, count in missing_values.items()
        if count > 0
    }

    print("[4/12] Checking exact duplicate rows...")
    report["exact_duplicate_rows"] = int(df.duplicated().sum())

    print("[5/12] Checking duplicate transaction records...")
    transaction_identity_columns = [
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

    report["duplicate_transaction_records"] = int(
        df.duplicated(subset=transaction_identity_columns).sum()
    )

    print("[6/12] Checking labels...")
    unique_labels = sorted(
        df["is_laundering"].dropna().unique().tolist()
    )

    report["unique_labels"] = unique_labels

    report["invalid_labels"] = [
        label
        for label in unique_labels
        if label not in [0, 1]
    ]

    print("[7/12] Checking data types...")
    report["dtypes"] = {
        col: str(df[col].dtype)
        for col in REQUIRED_COLUMNS
    }

    print("[8/12] Checking transaction amounts...")

    for col in ["amount_received", "amount_paid"]:
        numeric_values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        report[f"{col}_non_numeric"] = int(
            numeric_values.isna().sum() - df[col].isna().sum()
        )

        report[f"{col}_negative"] = int(
            (numeric_values < 0).sum()
        )

    print("[9/12] Checking self-transactions...")
    report["self_transactions"] = int(
        (df["from_account_id"] == df["to_account_id"]).sum()
    )

    print("[10/12] Checking account IDs...")
    report["empty_from_account_ids"] = int(
        (df["from_account_id"].astype(str).str.strip() == "").sum()
    )

    report["empty_to_account_ids"] = int(
        (df["to_account_id"].astype(str).str.strip() == "").sum()
    )

    print("[11/12] Checking timestamps...")
    report["null_timestamps"] = int(
        df["timestamp"].isna().sum()
    )

    # This can be expensive, but it's still useful.
    report["timestamps_out_of_order"] = int(
        (df["timestamp"].diff().dt.total_seconds() < 0).sum()
    )

    print("[12/12] Checking class distribution...")
    label_counts = df["is_laundering"].value_counts().sort_index()

    report["label_distribution"] = {
        int(label): int(count)
        for label, count in label_counts.items()
    }

    report["laundering_ratio"] = float(
        df["is_laundering"].mean()
    )

    return report


def print_report(report: dict) -> None:
    print("\n")
    print("=" * 75)
    print("IBM AML DATA VALIDATION REPORT")
    print("=" * 75)

    print("\nDATASET")
    print(f"Rows:                {report['rows']:,}")
    print(f"Columns:             {report['columns']}")

    print("\nSCHEMA")
    if report["missing_required_columns"]:
        print(
            "Missing required columns:",
            report["missing_required_columns"]
        )
    else:
        print("Missing required columns: None")

    print("\nMISSING VALUES")
    if report["missing_values"]:
        for column, count in report["missing_values"].items():
            print(f"{column:<25} {count:,}")
    else:
        print("None")

    print("\nDUPLICATES")
    print(
        f"Exact duplicate rows:          "
        f"{report['exact_duplicate_rows']:,}"
    )
    print(
        f"Duplicate transaction records: "
        f"{report['duplicate_transaction_records']:,}"
    )

    print("\nLABELS")
    print(f"Unique labels:  {report['unique_labels']}")
    print(f"Invalid labels: {report['invalid_labels']}")

    print("\nAMOUNTS")

    print(
        f"Non-numeric received: {report['amount_received_non_numeric']:,}"
    )

    print(
        f"Negative received:    {report['amount_received_negative']:,}"
    )

    print(
        f"Non-numeric paid:     {report['amount_paid_non_numeric']:,}"
    )

    print(
        f"Negative paid:        {report['amount_paid_negative']:,}"
    )

    print("\nGRAPH-RELATED CHECKS")

    print(
        f"Self-transactions:    "
        f"{report['self_transactions']:,}"
    )

    print(
        f"Empty source IDs:     "
        f"{report['empty_from_account_ids']:,}"
    )

    print(
        f"Empty destination IDs:"
        f" {report['empty_to_account_ids']:,}"
    )

    print("\nTIMESTAMP")

    print(
        f"Null timestamps:      "
        f"{report['null_timestamps']:,}"
    )

    print(
        f"Out-of-order rows:    "
        f"{report['timestamps_out_of_order']:,}"
    )

    print("\nCLASS DISTRIBUTION")

    for label, count in report["label_distribution"].items():
        print(f"Label {label}: {count:,}")

    print(
        f"Laundering ratio: "
        f"{report['laundering_ratio']:.6%}"
    )

    print("\n" + "=" * 75)
    print("VALIDATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":

    total_start = time.perf_counter()

    project_root = Path(__file__).resolve().parents[2]

    sys.path.insert(
        0,
        str(project_root / "src")
    )

    from ingestion.loader import load_transactions

    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/HI-Small_Trans.csv"
    )

    print("=" * 75)
    print("IBM AML DATA VALIDATION")
    print("=" * 75)

    print(f"\nDataset:")
    print(csv_path)

    print("\nLoading dataset...")
    print("This may take a while because the CSV contains ~5 million rows.\n")

    load_start = time.perf_counter()

    transactions = load_transactions(csv_path)

    load_time = time.perf_counter() - load_start

    print(
        f"\nDataset loaded successfully "
        f"in {load_time:.2f} seconds."
    )

    print(
        f"Memory usage: "
        f"{transactions.memory_usage(deep=True).sum() / (1024 ** 3):.2f} GB"
    )

    validation_start = time.perf_counter()

    report = validate_dataframe(transactions)

    validation_time = time.perf_counter() - validation_start

    print(
        f"\nValidation finished in "
        f"{validation_time:.2f} seconds."
    )

    print_report(report)

    total_time = time.perf_counter() - total_start

    print(
        f"\nTotal runtime: "
        f"{total_time:.2f} seconds."
    )