"""
Phase 1 — Data Ingestion

Loads the IBM Transactions for Anti-Money Laundering (AML) dataset and
produces a summary profile used to sanity-check the data before graph
construction (Phase 3).

Dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
Variants: {HI,LI}-{Small,Medium,Large}_Trans.csv
    HI = Higher Illicit ratio, LI = Lower Illicit ratio
Recommended for a first pass: HI-Small_Trans.csv (smallest, richest in
positive examples — best for fast iteration during development).

Raw transaction CSV has 11 columns:
    Timestamp, From Bank, Account, To Bank, Account.1,
    Amount Received, Receiving Currency, Amount Paid, Payment Currency,
    Payment Format, Is Laundering
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]

RENAME_MAP = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}


@dataclass
class DatasetProfile:
    n_transactions: int
    n_accounts: int
    n_banks: int
    n_laundering: int
    laundering_ratio: float
    date_range: tuple
    payment_formats: dict
    top_currencies: dict


def load_transactions(path: str | Path) -> pd.DataFrame:
    """
    Load a raw IBM-AML transaction CSV and normalize it for downstream use.

    Parameters
    ----------
    path : str | Path
        Path to a *_Trans.csv file (e.g. data/raw/HI-Small_Trans.csv).

    Returns
    -------
    pd.DataFrame
        Snake_case columns, parsed timestamps, and two new columns —
        `from_account_id` / `to_account_id` — that are GLOBALLY unique
        account identifiers.

    Notes
    -----
    The raw `Account` column is only unique *within* a bank (two different
    banks can both have an account numbered e.g. "800180E10"). If you build
    the graph directly off the raw `Account` column you will silently merge
    unrelated accounts from different banks into the same node. This
    function builds `{bank}_{account}` composite IDs specifically to avoid
    that — always use `from_account_id` / `to_account_id` from here on,
    not the raw `from_account` / `to_account` columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Download the dataset from Kaggle "
            "(ealtman2019/ibm-transactions-for-anti-money-laundering-aml), "
            "unzip it, and place the *_Trans.csv file under data/raw/."
        )

    df = pd.read_csv(path)

    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Unexpected schema for {path.name} — missing columns: {missing}. "
            f"Got columns: {list(df.columns)}. "
            "This loader expects the standard IBM-AML *_Trans.csv layout — "
            "if you're pointing this at a different file (e.g. the pattern "
            "file), use a different loader."
        )

    df = df.rename(columns=RENAME_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_laundering"] = df["is_laundering"].astype(int)

    # globally-unique node IDs — see docstring note above
    df["from_account_id"] = (
        df["from_bank"].astype(str) + "_" + df["from_account"].astype(str)
    )
    df["to_account_id"] = (
        df["to_bank"].astype(str) + "_" + df["to_account"].astype(str)
    )

    return df


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """
    Compute the summary statistics you need before touching graph
    construction: node/edge counts, class imbalance, time span, and
    the categorical distributions that will matter for feature engineering.
    """
    accounts = pd.unique(
        pd.concat([df["from_account_id"], df["to_account_id"]], ignore_index=True)
    )
    banks = pd.unique(
        pd.concat([df["from_bank"], df["to_bank"]], ignore_index=True)
    )

    return DatasetProfile(
        n_transactions=len(df),
        n_accounts=len(accounts),
        n_banks=len(banks),
        n_laundering=int(df["is_laundering"].sum()),
        laundering_ratio=float(df["is_laundering"].mean()),
        date_range=(df["timestamp"].min(), df["timestamp"].max()),
        payment_formats=df["payment_format"].value_counts().to_dict(),
        top_currencies=df["payment_currency"].value_counts().head(5).to_dict(),
    )


def print_profile(profile: DatasetProfile) -> None:
    print(f"Transactions:      {profile.n_transactions:,}")
    print(f"Unique accounts:   {profile.n_accounts:,}  (these become graph nodes)")
    print(f"Unique banks:      {profile.n_banks:,}")
    print(
        f"Laundering txns:   {profile.n_laundering:,} "
        f"({profile.laundering_ratio:.4%} of all transactions)"
    )
    print(f"Date range:        {profile.date_range[0]} -> {profile.date_range[1]}")
    print(f"Payment formats:   {profile.payment_formats}")
    print(f"Top currencies:    {profile.top_currencies}")


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/HI-Small_Trans.csv"
    transactions = load_transactions(csv_path)
    print_profile(profile_dataset(transactions))
