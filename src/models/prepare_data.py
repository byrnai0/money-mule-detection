from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "training_graph.pt"
)

RANDOM_STATE = 42


def main() -> None:

    start = time.perf_counter()

    print("=" * 75)
    print("PHASE 5A — TRAINING DATA PREPARATION")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load graph
    # ---------------------------------------------------------

    print("\n[1/7] Loading graph...")

    graph_payload = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )

    graph = graph_payload["graph"]
    account_list = graph_payload["account_list"]

    print(
        f"Graph nodes: {graph.num_nodes:,}"
    )

    print(
        f"Graph edges: "
        f"{graph.edge_index.shape[1]:,}"
    )

    # ---------------------------------------------------------
    # 2. Load features + labels
    # ---------------------------------------------------------

    print("\n[2/7] Loading features and labels...")

    features = pd.read_csv(FEATURE_FILE)
    labels = pd.read_csv(LABEL_FILE)

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Label rows: {len(labels):,}"
    )

    # ---------------------------------------------------------
    # 3. Align with graph account ordering
    # ---------------------------------------------------------

    print(
        "\n[3/7] Aligning accounts with graph node ordering..."
    )

    feature_index = features.set_index(
        "account_id"
    )

    label_index = labels.set_index(
        "account_id"
    )

    graph_accounts = pd.Index(
        account_list,
        name="account_id",
    )

    missing_features = (
        graph_accounts
        .difference(feature_index.index)
    )

    missing_labels = (
        graph_accounts
        .difference(label_index.index)
    )

    if len(missing_features) > 0:
        raise RuntimeError(
            f"{len(missing_features)} graph accounts "
            "have no features."
        )

    if len(missing_labels) > 0:
        raise RuntimeError(
            f"{len(missing_labels)} graph accounts "
            "have no labels."
        )

    X_df = feature_index.loc[
        graph_accounts
    ]

    y_series = label_index.loc[
        graph_accounts,
        "is_illicit",
    ]

    X = X_df.to_numpy(
        dtype=np.float32
    )

    y = y_series.to_numpy(
        dtype=np.int64
    )

    print(
        f"Aligned feature matrix: "
        f"{X.shape}"
    )

    print(
        f"Labels: {y.shape}"
    )

    # ---------------------------------------------------------
    # 4. Stratified 80/10/10 split
    # ---------------------------------------------------------

    print(
        "\n[4/7] Creating stratified train/val/test split..."
    )

    indices = np.arange(
        len(y)
    )

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=y[temp_idx],
        random_state=RANDOM_STATE,
    )

    train_mask = np.zeros(
        len(y),
        dtype=bool,
    )

    val_mask = np.zeros(
        len(y),
        dtype=bool,
    )

    test_mask = np.zeros(
        len(y),
        dtype=bool,
    )

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(
        f"Train: {train_mask.sum():,}"
    )

    print(
        f"Validation: {val_mask.sum():,}"
    )

    print(
        f"Test: {test_mask.sum():,}"
    )

    # ---------------------------------------------------------
    # 5. Fit scaler ONLY on training accounts
    # ---------------------------------------------------------

    print(
        "\n[5/7] Standardizing features using TRAIN only..."
    )

    scaler = StandardScaler()

    X_train = X[train_idx]

    scaler.fit(
        X_train
    )

    X_scaled = scaler.transform(
        X
    ).astype(
        np.float32
    )

    # ---------------------------------------------------------
    # 6. Create PyG Data object
    # ---------------------------------------------------------

    print(
        "\n[6/7] Creating training-ready PyG graph..."
    )

    data = Data(
        x=torch.from_numpy(
            X_scaled.copy()
        ),

        edge_index=graph.edge_index,

        y=torch.from_numpy(
            y.copy()
        ),

        train_mask=torch.from_numpy(
            train_mask.copy()
        ),

        val_mask=torch.from_numpy(
            val_mask.copy()
        ),

        test_mask=torch.from_numpy(
            test_mask.copy()
        ),

        num_nodes=len(y),
    )

    # Preserve transaction information.
    if hasattr(graph, "edge_label"):
        data.edge_label = graph.edge_label

    if hasattr(graph, "amount_received"):
        data.amount_received = graph.amount_received

    if hasattr(graph, "amount_paid"):
        data.amount_paid = graph.amount_paid

    if hasattr(graph, "timestamp"):
        data.timestamp = graph.timestamp

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------

    print(
        "\n[7/7] Saving training graph..."
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "data": data,
            "account_list": account_list,
            "scaler": scaler,
        },
        OUTPUT_FILE,
    )

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 75)
    print("PHASE 5A COMPLETE")
    print("=" * 75)

    print(
        f"\nNodes:      {data.num_nodes:,}"
    )

    print(
        f"Edges:      {data.edge_index.shape[1]:,}"
    )

    print(
        f"Features:   {data.x.shape[1]}"
    )

    print(
        f"Train:      {int(data.train_mask.sum()):,}"
    )

    print(
        f"Validation: {int(data.val_mask.sum()):,}"
    )

    print(
        f"Test:       {int(data.test_mask.sum()):,}"
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