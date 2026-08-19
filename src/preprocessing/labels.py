from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

print("=" * 75)
print("ACCOUNT-LEVEL LABEL DERIVATION")
print("=" * 75)
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
    / "account_labels.csv"
)


def derive_account_labels() -> None:

    start = time.perf_counter()


    print(f"\nInput:")
    print(INPUT_FILE)

    print(f"\nOutput:")
    print(OUTPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed transaction file not found:\n{INPUT_FILE}"
        )

    # -----------------------------------------------------
    # 1. Load only the columns we need
    # -----------------------------------------------------

    print("\n[1/5] Loading transaction labels...")

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "from_bank",
            "from_account",
            "to_bank",
            "to_account",
            "is_laundering",
        ],
    )

    print(
        f"Transactions loaded: {len(df):,}"
    )

    # -----------------------------------------------------
    # 2. Recreate globally unique account IDs
    # -----------------------------------------------------

    print(
        "\n[2/5] Creating globally unique account IDs..."
    )

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
    # 3. Extract accounts involved in laundering
    # -----------------------------------------------------

    print(
        "\n[3/5] Deriving source/destination illicit flags..."
    )

    illicit_transactions = df[
        df["is_laundering"] == 1
    ]

    print(
        f"Laundering transactions: "
        f"{len(illicit_transactions):,}"
    )

    illicit_sources = set(
        illicit_transactions["from_account_id"]
    )

    illicit_destinations = set(
        illicit_transactions["to_account_id"]
    )

    print(
        f"Accounts involved as illicit sources: "
        f"{len(illicit_sources):,}"
    )

    print(
        f"Accounts involved as illicit destinations: "
        f"{len(illicit_destinations):,}"
    )

    # -----------------------------------------------------
    # 4. Build complete account list
    # -----------------------------------------------------

    print(
        "\n[4/5] Building account-level label table..."
    )

    all_accounts = pd.unique(
        pd.concat(
            [
                df["from_account_id"],
                df["to_account_id"],
            ],
            ignore_index=True,
        )
    )

    labels = pd.DataFrame(
        {
            "account_id": all_accounts
        }
    )

    labels["is_illicit_source"] = (
        labels["account_id"]
        .isin(illicit_sources)
        .astype("int8")
    )

    labels["is_illicit_dest"] = (
        labels["account_id"]
        .isin(illicit_destinations)
        .astype("int8")
    )

    labels["is_illicit"] = (
        (
            labels["is_illicit_source"]
            == 1
        )
        |
        (
            labels["is_illicit_dest"]
            == 1
        )
    ).astype("int8")

    # -----------------------------------------------------
    # 5. Save + verification
    # -----------------------------------------------------

    print(
        "\n[5/5] Saving and verifying labels..."
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    total_accounts = len(labels)

    source_count = int(
        labels["is_illicit_source"].sum()
    )

    destination_count = int(
        labels["is_illicit_dest"].sum()
    )

    illicit_count = int(
        labels["is_illicit"].sum()
    )

    source_only = int(
        (
            (labels["is_illicit_source"] == 1)
            &
            (labels["is_illicit_dest"] == 0)
        ).sum()
    )

    destination_only = int(
        (
            (labels["is_illicit_source"] == 0)
            &
            (labels["is_illicit_dest"] == 1)
        ).sum()
    )

    both = int(
        (
            (labels["is_illicit_source"] == 1)
            &
            (labels["is_illicit_dest"] == 1)
        ).sum()
    )

    illicit_ratio = (
        illicit_count / total_accounts
        if total_accounts
        else 0
    )

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 75)
    print("ACCOUNT LABEL DERIVATION COMPLETE")
    print("=" * 75)

    print(
        f"\nTotal accounts:                 "
        f"{total_accounts:,}"
    )

    print(
        f"Illicit source accounts:       "
        f"{source_count:,}"
    )

    print(
        f"Illicit destination accounts:  "
        f"{destination_count:,}"
    )

    print(
        f"Illicit accounts (combined):   "
        f"{illicit_count:,}"
    )

    print(
        f"Source only:                    "
        f"{source_only:,}"
    )

    print(
        f"Destination only:               "
        f"{destination_only:,}"
    )

    print(
        f"Both source + destination:      "
        f"{both:,}"
    )

    print(
        f"Account illicit ratio:          "
        f"{illicit_ratio:.6%}"
    )

    print(
        f"\nOutput saved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nRuntime: {elapsed:.2f} seconds"
    )

    print("=" * 75)


if __name__ == "__main__":
    derive_account_labels()