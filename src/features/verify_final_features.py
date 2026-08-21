from pathlib import Path

import numpy as np
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

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "ibm_aml_graph.pt"
)


def main():

    print("=" * 75)
    print("PHASE 4 — FINAL FEATURE VERIFICATION")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    print("\n[1/6] Loading final feature matrix...")

    features = pd.read_csv(FEATURE_FILE)

    print(
        f"Rows: {len(features):,}"
    )

    print(
        f"Columns: {len(features.columns):,}"
    )

    # ---------------------------------------------------------
    # Basic structure
    # ---------------------------------------------------------

    print("\n[2/6] Checking structure...")

    if "account_id" not in features.columns:
        raise RuntimeError(
            "account_id column is missing."
        )

    if features["account_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate account IDs found."
        )

    feature_columns = [
        col
        for col in features.columns
        if col != "account_id"
    ]

    print(
        f"Feature count: {len(feature_columns)}"
    )

    # ---------------------------------------------------------
    # Numerical checks
    # ---------------------------------------------------------

    print("\n[3/6] Checking numerical values...")

    numeric = features[feature_columns]

    nan_count = int(
        numeric.isna().sum().sum()
    )

    inf_count = int(
        np.isinf(
            numeric.to_numpy()
        ).sum()
    )

    print(
        f"NaN values:       {nan_count}"
    )

    print(
        f"Infinite values:  {inf_count}"
    )

    if nan_count != 0:
        raise RuntimeError(
            "NaN values exist in final features."
        )

    if inf_count != 0:
        raise RuntimeError(
            "Infinite values exist in final features."
        )

    # ---------------------------------------------------------
    # Target leakage check
    # ---------------------------------------------------------

    print("\n[4/6] Checking target leakage...")

    forbidden = {
        "is_illicit",
        "is_illicit_source",
        "is_illicit_dest",
        "is_laundering",
    }

    leakage_columns = (
        forbidden
        .intersection(
            set(features.columns)
        )
    )

    print(
        f"Target columns found: {leakage_columns}"
    )

    if leakage_columns:
        raise RuntimeError(
            "Target leakage detected."
        )

    # ---------------------------------------------------------
    # Label alignment
    # ---------------------------------------------------------

    print("\n[5/6] Checking label alignment...")

    labels = pd.read_csv(
        LABEL_FILE,
        usecols=[
            "account_id",
            "is_illicit",
        ],
    )

    feature_accounts = set(
        features["account_id"]
    )

    label_accounts = set(
        labels["account_id"]
    )

    missing_from_labels = (
        feature_accounts
        - label_accounts
    )

    missing_from_features = (
        label_accounts
        - feature_accounts
    )

    print(
        f"Feature accounts: {len(feature_accounts):,}"
    )

    print(
        f"Label accounts:   {len(label_accounts):,}"
    )

    print(
        f"Missing labels:    {len(missing_from_labels):,}"
    )

    print(
        f"Missing features:  {len(missing_from_features):,}"
    )

    if missing_from_labels:
        raise RuntimeError(
            "Some feature accounts have no label."
        )

    if missing_from_features:
        raise RuntimeError(
            "Some labelled accounts have no features."
        )

    # ---------------------------------------------------------
    # Graph alignment
    # ---------------------------------------------------------

    print("\n[6/6] Checking graph account count...")

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph not found:\n{GRAPH_FILE}"
        )

    import torch

    payload = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )

    accounts = payload["account_list"]

    graph_accounts = set(accounts)

    missing_from_graph = (
        feature_accounts
        - graph_accounts
    )

    missing_from_feature_matrix = (
        graph_accounts
        - feature_accounts
    )

    print(
        f"Graph accounts:        {len(graph_accounts):,}"
    )

    print(
        f"Missing from graph:     "
        f"{len(missing_from_graph):,}"
    )

    print(
        f"Missing from features:  "
        f"{len(missing_from_feature_matrix):,}"
    )

    if missing_from_graph:
        raise RuntimeError(
            "Some feature accounts are missing from graph."
        )

    if missing_from_feature_matrix:
        raise RuntimeError(
            "Some graph accounts are missing features."
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("PHASE 4 FINAL VERIFICATION PASSED")
    print("=" * 75)

    print(
        f"\nAccounts:        {len(features):,}"
    )

    print(
        f"Features:        {len(feature_columns):,}"
    )

    print(
        f"Labels aligned:  YES"
    )

    print(
        f"Graph aligned:   YES"
    )

    print(
        f"NaN:             {nan_count}"
    )

    print(
        f"Infinity:        {inf_count}"
    )

    print(
        f"Target leakage:  NONE"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()