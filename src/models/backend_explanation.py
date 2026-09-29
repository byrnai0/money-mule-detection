from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from individual_explanation import (
    GRAPH_PATH,
    OUTPUT_DIR,
    load_feature_names,
    pretty_feature_name,
    risk_level,
)

from subgraph_explanation import (
    load_graph,
    load_models,
    extract_subgraph,
    get_neighbors,
    calculate_neighbor_impacts,
    get_transaction_details,
)


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "graphs"
    / "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "backend_explanations.json"
)

TOP_FEATURES = 5
TOP_NEIGHBORS = 5


# ============================================================
# FEATURE EXPLANATION
# ============================================================

def calculate_feature_explanations(
    models,
    local_x,
    local_edge_index,
    target_local_index,
    feature_names,
):
    """
    Calculate account-specific feature impacts.
    """

    with torch.no_grad():

        baseline_scores = []

        for name, model in models.items():

            logits = model(
                local_x,
                local_edge_index,
            )

            probability = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            baseline_scores.append(
                model_weight(name)
                * probability[
                    target_local_index
                ].item()
            )

        baseline = sum(
            baseline_scores
        )

    results = []

    for feature_index in range(
        local_x.shape[1]
    ):

        perturbed_x = local_x.clone()

        original_value = float(
            perturbed_x[
                target_local_index,
                feature_index,
            ].item()
        )

        # Standardized mean.
        perturbed_x[
            target_local_index,
            feature_index,
        ] = 0.0

        with torch.no_grad():

            score = 0.0

            for name, model in models.items():

                logits = model(
                    perturbed_x,
                    local_edge_index,
                )

                probability = torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]

                score += (
                    model_weight(name)
                    * probability[
                        target_local_index
                    ].item()
                )

        impact = baseline - score

        results.append(
            {
                "feature_index": feature_index,
                "feature": feature_names[
                    feature_index
                ],
                "display_name": pretty_feature_name(
                    feature_names[
                        feature_index
                    ]
                ),
                "original_value": original_value,
                "perturbed_score": float(score),
                "impact": float(impact),
                "absolute_impact": abs(
                    float(impact)
                ),
            }
        )

    results.sort(
        key=lambda x: x["absolute_impact"],
        reverse=True,
    )

    return baseline, results[:TOP_FEATURES]


# ============================================================
# MODEL WEIGHTS
# ============================================================

def model_weight(name: str) -> float:

    weights = {
        "sage": 0.78,
        "gatv2": 0.19,
        "gin": 0.03,
    }

    return weights[name]


# ============================================================
# BUILD EXPLANATION
# ============================================================

def build_backend_explanation(
    account_index: int,
):

    graph, account_list = load_graph()

    account_id = str(
        account_list[account_index]
    )

    feature_names = load_feature_names(
        graph.x.shape[1]
    )

    models = load_models(
        graph.x.shape[1]
    )

    # --------------------------------------------------------
    # Local graph
    # --------------------------------------------------------

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

    local_x = local_x.to(
        next(
            iter(models.values())
        ).parameters().__next__().device
    )

    local_edge_index = (
        local_edge_index.to(
            local_x.device
        )
    )

    # --------------------------------------------------------
    # Feature explanation
    # --------------------------------------------------------

    baseline_score, feature_results = (
        calculate_feature_explanations(
            models,
            local_x,
            local_edge_index,
            target_local_index,
            feature_names,
        )
    )

    # --------------------------------------------------------
    # Graph explanation
    # --------------------------------------------------------

    neighbors = get_neighbors(
        subset,
        local_edge_index,
        target_local_index,
    )

    _, neighbor_results = (
        calculate_neighbor_impacts(
            models,
            local_x,
            local_edge_index,
            target_local_index,
            neighbors,
        )
    )

    neighbor_results.sort(
        key=lambda x: x["absolute_impact"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Feature JSON
    # --------------------------------------------------------

    feature_output = []

    for item in feature_results:

        if item["impact"] > 0:
            direction = "positive"

        elif item["impact"] < 0:
            direction = "negative"

        else:
            direction = "neutral"

        feature_output.append(
            {
                "feature": item["feature"],
                "display_name": item[
                    "display_name"
                ],
                "original_value": item[
                    "original_value"
                ],
                "perturbed_score": item[
                    "perturbed_score"
                ],
                "impact": item["impact"],
                "direction": direction,
            }
        )

    # --------------------------------------------------------
    # Neighbor JSON
    # --------------------------------------------------------

    graph_output = []
    transaction_output = []

    for item in neighbor_results[
        :TOP_NEIGHBORS
    ]:

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

        transactions = (
            get_transaction_details(
                graph,
                subset,
                edge_mask,
                account_index,
                global_neighbor,
            )
        )

        graph_output.append(
            {
                "neighbor_id": neighbor_id,
                "neighbor_index": int(
                    global_neighbor
                ),
                "impact": item[
                    "score_change"
                ],
                "absolute_impact": item[
                    "absolute_impact"
                ],
                "score_after_removal": item[
                    "perturbed_score"
                ],
                "transaction_count": len(
                    transactions
                ),
            }
        )

        for transaction in transactions:

            transaction_output.append(
                {
                    "neighbor_id": neighbor_id,
                    "edge_index": int(
                        transaction[
                            "edge_index"
                        ]
                    ),
                    "amount_paid": transaction[
                        "amount_paid"
                    ],
                    "amount_received": transaction[
                        "amount_received"
                    ],
                    "edge_label": transaction[
                        "edge_label"
                    ],
                    "timestamp": transaction[
                        "timestamp"
                    ],
                }
            )

    # --------------------------------------------------------
    # Final object
    # --------------------------------------------------------

    explanation = {
        "account_id": account_id,
        "account_index": int(
            account_index
        ),
        "risk_score": float(
            baseline_score
        ),
        "risk_level": risk_level(
            baseline_score
        ),
        "model_weights": {
            "graphsage": 0.78,
            "gatv2": 0.19,
            "gin": 0.03,
        },
        "feature_reasons": feature_output,
        "graph_reasons": graph_output,
        "transactions": transaction_output,
        "interpretation_note": (
            "Impacts represent changes in the "
            "trained model's prediction after "
            "feature or neighbor perturbation. "
            "They do not establish real-world "
            "causation or illicit activity."
        ),
    }

    return explanation


# ============================================================
# MAIN
# ============================================================

def main(account_index: int):

    print("=" * 60)
    print("PHASE 8.5 — BACKEND EXPLANATION OUTPUT")
    print("=" * 60)

    print(
        f"Generating explanation for node "
        f"{account_index}..."
    )

    explanation = (
        build_backend_explanation(
            account_index
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep one file containing all generated accounts.
    if OUTPUT_FILE.exists():

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            existing = json.load(f)

    else:

        existing = {}

    existing[
        explanation["account_id"]
    ] = explanation

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            existing,
            f,
            indent=2,
        )

    print(
        f"\nAccount: "
        f"{explanation['account_id']}"
    )

    print(
        f"Risk score: "
        f"{explanation['risk_score']:.6f}"
    )

    print(
        f"Risk level: "
        f"{explanation['risk_level']}"
    )

    print(
        f"Feature explanations: "
        f"{len(explanation['feature_reasons'])}"
    )

    print(
        f"Graph explanations: "
        f"{len(explanation['graph_reasons'])}"
    )

    print(
        f"Transactions: "
        f"{len(explanation['transactions'])}"
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

    print(
        "\nPHASE 8E COMPLETE"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--account",
        type=int,
    )

    group.add_argument(
        "--account-id",
        type=str,
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

    main(
        account_index
    )