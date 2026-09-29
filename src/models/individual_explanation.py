from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph

from architectures import build_model


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = PROJECT_ROOT / "data" / "graphs" / "training_graph.pt"
MODEL_DIR = PROJECT_ROOT / "data" / "graphs" / "models" / "sampled"
FEATURE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "feature_manifest.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "graphs" / "evaluation"
OUTPUT_PATH = OUTPUT_DIR / "individual_explanations.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Number of graph hops used for the local explanation.
NUM_HOPS = 2

# How many features to show in the final explanation.
TOP_K = 5


# ============================================================
# FEATURE NAMES
# ============================================================

def load_feature_names(num_features: int) -> list[str]:
    """
    Load feature names from the feature manifest.

    Falls back to generic names if the manifest is unavailable.
    """

    if FEATURE_MANIFEST.exists():

        manifest = pd.read_csv(FEATURE_MANIFEST)

        # Try common column names.
        for column in ["feature", "feature_name", "name"]:
            if column in manifest.columns:
                names = manifest[column].astype(str).tolist()

                if len(names) == num_features:
                    return names

    return [f"feature_{i}" for i in range(num_features)]


# ============================================================
# MODEL LOADING
# ============================================================

def load_models(num_features: int):
    """
    Load the three frozen sampled models.
    """

    models = {}

    for name, filename in MODEL_FILES.items():

        model = build_model(
            name=name,
            in_channels=num_features,
            hidden_channels=64,
            dropout=0.30,
        )

        path = MODEL_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )

        # Support both raw state_dict and checkpoint dictionaries.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        model.to(DEVICE)
        model.eval()

        models[name] = model

        print(f"Loaded {name}: {path.name}")

    return models


# ============================================================
# RISK SCORE
# ============================================================

@torch.no_grad()
def ensemble_score(
    models,
    x,
    edge_index,
    target_index: int,
) -> float:
    """
    Calculate the frozen weighted ensemble probability
    for one target node.
    """

    scores = []

    for name, model in models.items():

        logits = model(
            x,
            edge_index,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        score = probabilities[target_index].item()

        scores.append(
            WEIGHTS[name] * score
        )

    return float(sum(scores))


# ============================================================
# LOCAL SUBGRAPH
# ============================================================

def extract_local_subgraph(
    graph,
    account_index: int,
    num_hops: int = NUM_HOPS,
):
    """
    Extract a small k-hop neighborhood around the account.

    The returned node mapping keeps track of where the
    original account sits inside the local subgraph.
    """

    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=account_index,
        num_hops=num_hops,
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
    )


# ============================================================
# FEATURE IMPACT
# ============================================================

def calculate_feature_impacts(
    models,
    local_x,
    local_edge_index,
    target_local_index,
):
    """
    Perturb each feature of the target account.

    The feature is replaced by zero in the standardized
    feature space.

    The resulting drop in risk score represents how much
    the model relied on that feature for this account.
    """

    baseline_score = ensemble_score(
        models,
        local_x,
        local_edge_index,
        target_local_index,
    )

    impacts = []

    num_features = local_x.shape[1]

    for feature_idx in range(num_features):

        perturbed_x = local_x.clone()

        original_value = perturbed_x[
            target_local_index,
            feature_idx,
        ].item()

        # Features were standardized during Phase 5,
        # so zero represents the training-set mean.
        perturbed_x[
            target_local_index,
            feature_idx,
        ] = 0.0

        perturbed_score = ensemble_score(
            models,
            perturbed_x,
            local_edge_index,
            target_local_index,
        )

        score_change = baseline_score - perturbed_score

        impacts.append(
            {
                "feature_index": feature_idx,
                "original_value": original_value,
                "perturbed_score": perturbed_score,
                "score_change": score_change,
                "absolute_impact": abs(score_change),
            }
        )

    return baseline_score, impacts


# ============================================================
# HUMAN READABLE FEATURE NAMES
# ============================================================

def pretty_feature_name(name: str) -> str:

    replacements = {
        "log1p_": "",
        "_": " ",
        "incoming": "incoming",
        "outgoing": "outgoing",
        "transaction": "transaction",
        "transactions": "transactions",
        "payment": "payment",
        "currency": "currency",
        "diversity": "diversity",
        "count": "count",
        "amount": "amount",
        "maximum": "maximum",
        "max": "maximum",
        "mean": "mean",
        "std": "standard deviation",
        "unique": "unique",
        "receiver": "receiver",
        "receivers": "receivers",
        "sender": "sender",
        "senders": "senders",
        "bank": "bank",
        "banks": "banks",
        "ratio": "ratio",
        "burst": "burst",
    }

    result = name

    for old, new in replacements.items():
        result = result.replace(old, new)

    return " ".join(result.split()).strip().capitalize()


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(score: float) -> str:

    if score >= 0.79:
        return "HIGH"

    if score >= 0.50:
        return "MEDIUM"

    return "LOW"


# ============================================================
# EXPLANATION
# ============================================================

def build_explanation(
    account_id,
    baseline_score,
    impacts,
    feature_names,
):
    """
    Build a human-readable explanation from the strongest
    feature perturbations.
    """

    impacts_sorted = sorted(
        impacts,
        key=lambda x: x["absolute_impact"],
        reverse=True,
    )

    top = impacts_sorted[:TOP_K]

    reasons = []

    for item in top:

        feature_name = feature_names[
            item["feature_index"]
        ]

        readable_name = pretty_feature_name(
            feature_name
        )

        change = item["score_change"]

        if change > 0:
            direction = "increased risk"
        elif change < 0:
            direction = "reduced risk"
        else:
            direction = "had little effect"

        reasons.append(
            f"{readable_name} ({direction}, "
            f"score change {change:+.4f})"
        )

    return reasons, top


# ============================================================
# MAIN
# ============================================================

def explain_account(account_index: int):

    print("=" * 60)
    print("PHASE 8.2 — INDIVIDUAL ACCOUNT EXPLANATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    print("\nLoading graph...")

    bundle = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    graph = bundle["data"]
    account_list = bundle["account_list"]
    scaler = bundle["scaler"]

    print(f"Accounts: {graph.x.shape[0]}")
    print(f"Features: {graph.x.shape[1]}")
    print(f"Edges: {graph.edge_index.shape[1]}")

    if account_index < 0 or account_index >= len(account_list):
        raise ValueError(
            f"Account index must be between "
            f"0 and {graph.x.shape[0] - 1}"
        )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = load_feature_names(
        graph.x.shape[1]
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading models...")

    models = load_models(
        graph.x.shape[1]
    )

    # --------------------------------------------------------
    # Extract local graph
    # --------------------------------------------------------

    print(
        f"\nExtracting {NUM_HOPS}-hop neighborhood..."
    )

    (
        subset,
        local_x,
        local_edge_index,
        target_local_index,
    ) = extract_local_subgraph(
        graph,
        account_index,
    )

    print(
        f"Local nodes: {local_x.shape[0]}"
    )

    print(
        f"Local edges: {local_edge_index.shape[1]}"
    )

    # Move only the local graph to GPU.
    local_x = local_x.to(DEVICE)
    local_edge_index = local_edge_index.to(DEVICE)

    # --------------------------------------------------------
    # Feature perturbation
    # --------------------------------------------------------

    print("\nCalculating feature impacts...")

    baseline_score, impacts = calculate_feature_impacts(
        models,
        local_x,
        local_edge_index,
        target_local_index,
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    reasons, top = build_explanation(
        account_index,
        baseline_score,
        impacts,
        feature_names,
    )

    level = risk_level(
        baseline_score
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ACCOUNT EXPLANATION")
    print("=" * 60)

    print(
        f"Account index : {account_index}"
    )

    print(
    f"Account ID    : {account_list[account_index]}"
    )

    print(
        f"Risk score    : {baseline_score:.6f}"
    )

    print(
        f"Risk level    : {level}"
    )

    print("\nMain factors:")

    for i, reason in enumerate(reasons, 1):
        print(
            f"{i}. {reason}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for rank, item in enumerate(top, 1):

        feature_idx = item["feature_index"]

        rows.append(
            {
                "account_index": account_index,
                "risk_score": baseline_score,
                "risk_level": level,
                "feature_rank": rank,
                "feature_index": feature_idx,
                "feature": feature_names[feature_idx],
                "original_value": item["original_value"],
                "perturbed_score": item["perturbed_score"],
                "score_change": item["score_change"],
                "absolute_impact": item["absolute_impact"],
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved explanation:"
    )

    print(
        OUTPUT_PATH
    )

    print("\nPHASE 8.2 COMPLETE")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Explain an individual account prediction."
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--account",
        type=int,
        help="Account node index to explain.",
    )

    group.add_argument(
        "--account-id",
        type=str,
        help="Original account ID to explain.",
    )

    args = parser.parse_args()

    if args.account is not None:

        account_index = args.account

    else:

        bundle = torch.load(
            GRAPH_PATH,
            map_location="cpu",
            weights_only=False,
        )

        account_list = bundle["account_list"]

        account_id = str(args.account_id)

        account_strings = [
            str(x) for x in account_list
        ]

        if account_id not in account_strings:
            raise ValueError(
                f"Account ID '{account_id}' "
                f"was not found in account_list."
            )

        account_index = account_strings.index(
            account_id
        )

        print(
            f"Account ID '{account_id}' "
            f"maps to node index {account_index}"
        )

    explain_account(
        account_index
    )