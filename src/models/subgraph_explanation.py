from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph

from architectures import build_model


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "training_graph.pt"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "models"
    / "sampled"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "subgraph_explanations.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WEIGHTS = {
    "sage": 0.78,
    "gatv2": 0.19,
    "gin": 0.03,
}

MODEL_FILES = {
    "sage": "sage_sampled_model.pt",
    "gatv2": "gatv2_sampled_model.pt",
    "gin": "gin_sampled_model.pt",
}

NUM_HOPS = 2

TOP_NEIGHBORS = 10


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    bundle = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    graph = bundle["data"]
    account_list = bundle["account_list"]

    return graph, account_list


# ============================================================
# LOAD MODELS
# ============================================================

def load_models(num_features):

    models = {}

    for name, filename in MODEL_FILES.items():

        model = build_model(
            name=name,
            in_channels=num_features,
            hidden_channels=64,
            dropout=0.30,
        )

        model_path = MODEL_DIR / filename

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        model.to(DEVICE)
        model.eval()

        models[name] = model

        print(
            f"Loaded {name}: {filename}"
        )

    return models


# ============================================================
# ENSEMBLE SCORE
# ============================================================

@torch.no_grad()
def ensemble_score(
    models,
    x,
    edge_index,
    target_index,
):

    total_score = 0.0

    for name, model in models.items():

        logits = model(
            x,
            edge_index,
        )

        probability = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        score = probability[
            target_index
        ].item()

        total_score += (
            WEIGHTS[name] * score
        )

    return float(total_score)


# ============================================================
# TARGET SUBGRAPH
# ============================================================

def extract_subgraph(
    graph,
    account_index,
):

    (
        subset,
        sub_edge_index,
        mapping,
        edge_mask,
    ) = k_hop_subgraph(
        node_idx=account_index,
        num_hops=NUM_HOPS,
        edge_index=graph.edge_index,
        relabel_nodes=True,
    )

    local_x = graph.x[subset]

    target_local_index = mapping.item()

    return (
        subset,
        local_x,
        sub_edge_index,
        target_local_index,
        edge_mask,
    )


# ============================================================
# NEIGHBOR IDENTIFICATION
# ============================================================

def get_neighbors(
    subset,
    sub_edge_index,
    target_local_index,
):

    neighbors = set()

    source_nodes = sub_edge_index[0]
    target_nodes = sub_edge_index[1]

    for src, dst in zip(
        source_nodes.tolist(),
        target_nodes.tolist(),
    ):

        if src == target_local_index and dst != target_local_index:
            neighbors.add(dst)

        if dst == target_local_index and src != target_local_index:
            neighbors.add(src)

    return sorted(neighbors)


# ============================================================
# NEIGHBOR PERTURBATION
# ============================================================

def calculate_neighbor_impacts(
    models,
    local_x,
    local_edge_index,
    target_local_index,
    neighbors,
):

    baseline_score = ensemble_score(
        models,
        local_x,
        local_edge_index,
        target_local_index,
    )

    results = []

    for neighbor in neighbors:

        # Keep the target node.
        # Remove the neighbor's feature information
        # by replacing it with the standardized mean (0).
        perturbed_x = local_x.clone()

        perturbed_x[neighbor] = 0.0

        perturbed_score = ensemble_score(
            models,
            perturbed_x,
            local_edge_index,
            target_local_index,
        )

        score_change = (
            baseline_score
            - perturbed_score
        )

        results.append(
            {
                "neighbor_local_index": neighbor,
                "perturbed_score": perturbed_score,
                "score_change": score_change,
                "absolute_impact": abs(
                    score_change
                ),
            }
        )

    return baseline_score, results


# ============================================================
# TRANSACTION DETAILS
# ============================================================

def get_transaction_details(
    graph,
    subset,
    edge_mask,
    account_index,
    neighbor_global_index,
):

    original_edge_indices = (
        torch.nonzero(
            edge_mask,
            as_tuple=False,
        )
        .view(-1)
        .tolist()
    )

    rows = []

    for edge_idx in original_edge_indices:

        src = graph.edge_index[
            0,
            edge_idx,
        ].item()

        dst = graph.edge_index[
            1,
            edge_idx,
        ].item()

        if not (
            (
                src == account_index
                and dst == neighbor_global_index
            )
            or
            (
                dst == account_index
                and src == neighbor_global_index
            )
        ):
            continue

        rows.append(
            {
                "edge_index": edge_idx,
                "source_node": src,
                "target_node": dst,
                "edge_label": float(
                    graph.edge_label[
                        edge_idx
                    ].item()
                ),
                "amount_paid": float(
                    graph.amount_paid[
                        edge_idx
                    ].item()
                ),
                "amount_received": float(
                    graph.amount_received[
                        edge_idx
                    ].item()
                ),
                "timestamp": int(
                    graph.timestamp[
                        edge_idx
                    ].item()
                ),
            }
        )

    return rows


# ============================================================
# MAIN EXPLANATION
# ============================================================

def explain_account(account_index):

    print("=" * 60)
    print("PHASE 8.3 — GNN SUBGRAPH EXPLANATION")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    print("\nLoading graph...")

    graph, account_list = load_graph()

    print(
        f"Accounts: {graph.x.shape[0]}"
    )

    print(
        f"Edges: {graph.edge_index.shape[1]}"
    )

    print(
        f"Features: {graph.x.shape[1]}"
    )

    if (
        account_index < 0
        or account_index >= graph.x.shape[0]
    ):
        raise ValueError(
            f"Invalid account index: "
            f"{account_index}"
        )

    account_id = str(
        account_list[account_index]
    )

    print(
        f"Account ID: {account_id}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("\nLoading models...")

    models = load_models(
        graph.x.shape[1]
    )

    # --------------------------------------------------------
    # Subgraph
    # --------------------------------------------------------

    print(
        f"\nExtracting {NUM_HOPS}-hop subgraph..."
    )

    (
        subset,
        local_x,
        local_edge_index,
        target_local_index,
        edge_mask,
    ) = extract_subgraph(
        graph,
        account_index,
    )

    print(
        f"Local nodes: {local_x.shape[0]}"
    )

    print(
        f"Local edges: {local_edge_index.shape[1]}"
    )

    # --------------------------------------------------------
    # Neighbors
    # --------------------------------------------------------

    neighbors = get_neighbors(
        subset,
        local_edge_index,
        target_local_index,
    )

    print(
        f"Direct neighbors: {len(neighbors)}"
    )

    # Move local graph to GPU.
    local_x = local_x.to(DEVICE)
    local_edge_index = (
        local_edge_index.to(DEVICE)
    )

    # --------------------------------------------------------
    # Perturb neighbors
    # --------------------------------------------------------

    print(
        "\nCalculating neighbor influence..."
    )

    (
        baseline_score,
        impacts,
    ) = calculate_neighbor_impacts(
        models,
        local_x,
        local_edge_index,
        target_local_index,
        neighbors,
    )

    impacts.sort(
        key=lambda x: x["absolute_impact"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUBGRAPH EXPLANATION")
    print("=" * 60)

    print(
        f"Account ID  : {account_id}"
    )

    print(
        f"Risk score  : {baseline_score:.6f}"
    )

    print(
        f"Neighbors   : {len(neighbors)}"
    )

    print("\nTop influential neighbors:")

    rows = []

    for rank, item in enumerate(
        impacts[:TOP_NEIGHBORS],
        start=1,
    ):

        local_neighbor = (
            item["neighbor_local_index"]
        )

        global_neighbor = subset[
            local_neighbor
        ].item()

        neighbor_id = str(
            account_list[
                global_neighbor
            ]
        )

        transaction_rows = (
            get_transaction_details(
                graph,
                subset,
                edge_mask,
                account_index,
                global_neighbor,
            )
        )

        print(
            f"\n{rank}. "
            f"Neighbor: {neighbor_id}"
        )

        print(
            f"   Impact: "
            f"{item['score_change']:+.6f}"
        )

        print(
            f"   Score after removal: "
            f"{item['perturbed_score']:.6f}"
        )

        print(
            f"   Transactions: "
            f"{len(transaction_rows)}"
        )

        for transaction in transaction_rows[:5]:

            print(
                "      "
                f"edge={transaction['edge_index']} "
                f"paid={transaction['amount_paid']:.2f} "
                f"received={transaction['amount_received']:.2f} "
                f"label={transaction['edge_label']} "
                f"time={transaction['timestamp']}"
            )

        rows.append(
            {
                "account_index": account_index,
                "account_id": account_id,
                "baseline_risk_score": baseline_score,
                "neighbor_rank": rank,
                "neighbor_index": global_neighbor,
                "neighbor_id": neighbor_id,
                "score_after_removal": item[
                    "perturbed_score"
                ],
                "score_change": item[
                    "score_change"
                ],
                "absolute_impact": item[
                    "absolute_impact"
                ],
                "transaction_count": len(
                    transaction_rows
                ),
            }
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df = pd.DataFrame(rows)

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nPHASE 8.3 COMPLETE"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Explain GNN prediction using "
            "neighbor influence."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--account",
        type=int,
        help="Account node index.",
    )

    group.add_argument(
        "--account-id",
        type=str,
        help="Original account ID.",
    )

    args = parser.parse_args()

    # Resolve account ID.
    if args.account is not None:

        account_index = args.account

    else:

        bundle = torch.load(
            GRAPH_PATH,
            map_location="cpu",
            weights_only=False,
        )

        account_list = bundle[
            "account_list"
        ]

        account_strings = [
            str(x)
            for x in account_list
        ]

        account_id = str(
            args.account_id
        )

        if account_id not in account_strings:

            raise ValueError(
                f"Account ID '{account_id}' "
                f"was not found."
            )

        account_index = (
            account_strings.index(
                account_id
            )
        )

        print(
            f"Account ID '{account_id}' "
            f"maps to node index "
            f"{account_index}"
        )

    explain_account(
        account_index
    )