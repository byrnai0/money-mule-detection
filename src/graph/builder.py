from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

print("Graph nodes creation in process", flush=True)
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
    / "graphs"
    / "ibm_aml_graph.pt"
)


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
]


def build_graph() -> None:

    start = time.perf_counter()

    print("=" * 75)
    print("PHASE 3 — GRAPH CONSTRUCTION")
    print("=" * 75)

    print(f"\nInput dataset:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{INPUT_FILE}"
        )

    print("\n[1/7] Loading cleaned dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        usecols=REQUIRED_COLUMNS
    )

    print(f"Transactions loaded: {len(df):,}")

    # -----------------------------------------------------
    # 2. Create globally unique account IDs
    # -----------------------------------------------------

    print("\n[2/7] Creating globally unique account IDs...")

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
    # 3. Create account → node mapping
    # -----------------------------------------------------

    print("\n[3/7] Creating account-to-node mapping...")

    all_accounts = pd.unique(
        pd.concat(
            [
                df["from_account_id"],
                df["to_account_id"],
            ],
            ignore_index=True,
        )
    )

    account_to_node = pd.Series(
        np.arange(len(all_accounts), dtype=np.int64),
        index=all_accounts,
    )

    src_nodes = (
        df["from_account_id"]
        .map(account_to_node)
        .to_numpy(dtype=np.int64)
    )

    dst_nodes = (
        df["to_account_id"]
        .map(account_to_node)
        .to_numpy(dtype=np.int64)
    )

    num_nodes = len(all_accounts)

    print(f"Unique accounts / nodes: {num_nodes:,}")

    # -----------------------------------------------------
    # 4. Build directed edge index
    # -----------------------------------------------------

    print("\n[4/7] Building edge_index...")

    edge_index = torch.tensor(
        np.vstack(
            [
                src_nodes,
                dst_nodes,
            ]
        ),
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # 5. Preserve transaction information
    # -----------------------------------------------------

    print("\n[5/7] Preparing edge information...")

    timestamps = pd.to_datetime(
        df["timestamp"]
    )

    timestamp_seconds = (
        timestamps.astype("int64") // 10**9
    ).to_numpy(dtype=np.int64)

    edge_label = torch.tensor(
        df["is_laundering"].to_numpy(
            dtype=np.float32
        )
    )

    amount_received = torch.tensor(
        df["amount_received"].to_numpy(
            dtype=np.float32
        )
    )

    amount_paid = torch.tensor(
        df["amount_paid"].to_numpy(
            dtype=np.float32
        )
    )

    timestamp = torch.tensor(
        timestamp_seconds,
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # 6. Create PyG graph
    # -----------------------------------------------------

    print("\n[6/7] Creating PyTorch Geometric graph...")

    graph = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
    )

    graph.edge_label = edge_label
    graph.amount_received = amount_received
    graph.amount_paid = amount_paid
    graph.timestamp = timestamp

    # -----------------------------------------------------
    # 7. Save graph + account mapping
    # -----------------------------------------------------

    print("\n[7/7] Saving graph...")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "graph": graph,
            "account_list": list(all_accounts),
        },
        OUTPUT_FILE,
    )

    elapsed = time.perf_counter() - start

    laundering_edges = int(
        graph.edge_label.sum().item()
    )

    print("\n" + "=" * 75)
    print("PHASE 3 — GRAPH CONSTRUCTION COMPLETE")
    print("=" * 75)

    print(f"\nNodes:              {num_nodes:,}")
    print(f"Edges:              {graph.edge_index.shape[1]:,}")
    print(f"Laundering edges:   {laundering_edges:,}")

    print(
        f"Edge index shape:   "
        f"{tuple(graph.edge_index.shape)}"
    )

    print(f"Runtime:             {elapsed:.2f} seconds")

    print(
        f"\nGraph saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    build_graph()