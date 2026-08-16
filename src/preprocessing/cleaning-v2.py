from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd

print("CLEANING V2 STARTED", flush=True)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "HI-Small_Trans.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "HI-Small_Trans_clean.csv"
)


def clean_dataset(
    input_path: str | Path,
    output_path: str | Path,
) -> None:

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 75)
    print("PHASE 2 — DATA CLEANING")
    print("=" * 75)

    print(f"\nInput : {input_path}")
    print(f"Output: {output_path}")

    start = time.perf_counter()

    # -----------------------------------------------------
    # Load data through our existing loader
    # -----------------------------------------------------

    print("\n[1/5] Loading dataset...")

    sys.path.insert(
        0,
        str(PROJECT_ROOT / "src")
    )

    from ingestion.loader import load_transactions

    df = load_transactions(input_path)

    rows_before = len(df)
    laundering_before = int(df["is_laundering"].sum())

    print(f"Rows before cleaning: {rows_before:,}")

    # -----------------------------------------------------
    # Identify exact duplicates
    # -----------------------------------------------------

    print("[2/5] Detecting exact duplicate transactions...")

    duplicate_mask = df.duplicated(
        keep="first"
    )

    duplicates_removed = int(
        duplicate_mask.sum()
    )

    print(
        f"Exact duplicate records found: "
        f"{duplicates_removed:,}"
    )

    # -----------------------------------------------------
    # Remove only exact duplicate records
    # -----------------------------------------------------

    print("[3/5] Removing exact duplicates...")

    df_clean = df.loc[
        ~duplicate_mask
    ].copy()

    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    print("[4/5] Verifying cleaned dataset...")

    rows_after = len(df_clean)
    laundering_after = int(
        df_clean["is_laundering"].sum()
    )

    remaining_duplicates = int(
        df_clean.duplicated().sum()
    )

    print(f"Rows after cleaning: {rows_after:,}")
    print(
        f"Laundering before:  {laundering_before:,}"
    )
    print(
        f"Laundering after:   {laundering_after:,}"
    )
    print(
        f"Remaining exact duplicates: "
        f"{remaining_duplicates:,}"
    )

    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------

    if remaining_duplicates != 0:
        raise RuntimeError(
            "Duplicate rows still remain after cleaning."
        )

    if laundering_before != laundering_after:
        raise RuntimeError(
            "Number of laundering transactions changed "
            "during duplicate removal."
        )

    if rows_before - rows_after != duplicates_removed:
        raise RuntimeError(
            "Row-count verification failed."
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print("[5/5] Saving cleaned dataset...")

    df_clean.to_csv(
        output_path,
        index=False
    )

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 75)
    print("CLEANING COMPLETE")
    print("=" * 75)

    print(f"\nRows before:          {rows_before:,}")
    print(f"Duplicates removed:  {duplicates_removed:,}")
    print(f"Rows after:           {rows_after:,}")
    print(f"Laundering before:    {laundering_before:,}")
    print(f"Laundering after:     {laundering_after:,}")
    print(f"Output file:          {output_path}")
    print(f"Runtime:              {elapsed:.2f} seconds")
    print("=" * 75)


if __name__ == "__main__":

    input_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else DEFAULT_INPUT
    )

    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else DEFAULT_OUTPUT
    )

    clean_dataset(
        input_path,
        output_path
    )