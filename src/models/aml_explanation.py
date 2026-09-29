from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from individual_explanation import (
    GRAPH_PATH,
    OUTPUT_DIR,
    explain_account,
    load_feature_names,
)

from subgraph_explanation import (
    extract_subgraph,
    get_neighbors,
    calculate_neighbor_impacts,
    get_transaction_details,
    load_graph,
    load_models,
)


# ============================================================
# CONFIG
# ============================================================

INDIVIDUAL_OUTPUT = (
    OUTPUT_DIR / "individual_explanations.csv"
)

SUBGRAPH_OUTPUT = (
    OUTPUT_DIR / "subgraph_explanations.csv"
)

REPORT_DIR = (
    OUTPUT_DIR / "account_explanations"
)


# ============================================================
# FEATURE LANGUAGE
# ============================================================

FEATURE_DESCRIPTIONS = {

    "outgoing_log_amount_max":
        "outgoing transaction amount behavior",

    "outgoing_log_amount_std":
        "variation in outgoing transaction amounts",

    "outgoing_log_amount_mean":
        "average outgoing transaction amount",

    "incoming_log_amount_max":
        "maximum incoming transaction amount",

    "incoming_log_amount_mean":
        "average incoming transaction amount",

    "incoming_log_amount_std":
        "variation in incoming transaction amounts",

    "log1p_in_transaction_count":
        "incoming transaction activity",

    "log1p_out_transaction_count":
        "outgoing transaction activity",

    "log1p_unique_senders":
        "diversity of sending counterparties",

    "log1p_unique_receivers":
        "diversity of receiving counterparties",

    "log1p_unique_receiver_banks":
        "diversity of receiver banks",

    "log1p_incoming_payment_format_diversity":
        "incoming payment-format diversity",

    "log1p_outgoing_payment_format_diversity":
        "outgoing payment-format diversity",

    "log1p_receiving_currency_diversity":
        "receiving currency diversity",

    "log1p_payment_currency_diversity":
        "payment currency diversity",

    "activity_burst_ratio":
        "transaction activity burst behavior",

    "reciprocity_ratio":
        "transaction reciprocity",

    "counterparty_concentration":
        "counterparty concentration",

    "incoming_currency_abs_z_mean":
        "average incoming currency deviation",

    "incoming_currency_abs_z_max":
        "maximum incoming currency deviation",
}


# ============================================================
# FEATURE DESCRIPTION
# ============================================================

def describe_feature(feature_name: str) -> str:

    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[
            feature_name
        ]

    readable = feature_name

    if readable.startswith("log1p_"):
        readable = readable[6:]

    readable = readable.replace(
        "_",
        " ",
    )

    return readable


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
# FEATURE REASONS
# ============================================================

def build_feature_reasons(
    explanation_df,
):

    reasons = []

    for _, row in explanation_df.iterrows():

        feature = row["feature"]
        change = float(
            row["score_change"]
        )

        description = describe_feature(
            feature
        )

        magnitude = abs(change)

        if magnitude < 0.01:
            strength = "minor"

        elif magnitude < 0.05:
            strength = "moderate"

        elif magnitude < 0.15:
            strength = "substantial"

        else:
            strength = "strong"

        if change > 0:

            text = (
                f"{description.capitalize()} "
                f"was a {strength} positive contributor "
                f"to the model's predicted risk score."
            )

        elif change < 0:

            text = (
                f"{description.capitalize()} "
                f"had a {strength} opposing effect "
                f"on the model's predicted risk score."
            )

        else:

            text = (
                f"{description.capitalize()} "
                f"had little measurable effect "
                f"on the model's predicted risk score."
            )

        reasons.append(text)

    return reasons


# ============================================================
# GRAPH REASONS
# ============================================================

def build_graph_reasons(
    subgraph_df,
):

    reasons = []

    for _, row in subgraph_df.iterrows():

        neighbor = str(
            row["neighbor_id"]
        )

        change = float(
            row["score_change"]
        )

        transactions = int(
            row["transaction_count"]
        )

        if change > 0:

            effect = (
                "increased the model's predicted "
                "risk score"
            )

        elif change < 0:

            effect = (
                "reduced the model's predicted "
                "risk score"
            )

        else:

            effect = (
                "had little measurable effect on "
                "the model's predicted risk score"
            )

        reasons.append(
            f"Neighbor {neighbor} "
            f"{effect} when its feature information "
            f"was removed from the local graph "
            f"(score change {change:+.4f}); "
            f"{transactions} connecting transaction(s) "
            f"were identified."
        )

    return reasons


# ============================================================
# TRANSACTION SUMMARY
# ============================================================

def transaction_summary(
    graph,
    subset,
    edge_mask,
    account_index,
    neighbor_index,
):

    transactions = get_transaction_details(
        graph,
        subset,
        edge_mask,
        account_index,
        neighbor_index,
    )

    if not transactions:
        return []

    summaries = []

    for transaction in transactions:

        summaries.append(
            (
                f"Edge {transaction['edge_index']}: "
                f"paid={transaction['amount_paid']:.2f}, "
                f"received="
                f"{transaction['amount_received']:.2f}, "
                f"edge_label="
                f"{transaction['edge_label']}, "
                f"timestamp="
                f"{transaction['timestamp']}"
            )
        )

    return summaries


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    account_index,
):

    print(
        "\nGenerating human-readable explanation..."
    )

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    graph, account_list = load_graph()

    account_id = str(
        account_list[account_index]
    )

    # --------------------------------------------------------
    # Load feature explanation
    # --------------------------------------------------------

    if not INDIVIDUAL_OUTPUT.exists():

        raise FileNotFoundError(
            f"Missing individual explanation: "
            f"{INDIVIDUAL_OUTPUT}"
        )

    individual_df = pd.read_csv(
        INDIVIDUAL_OUTPUT
    )

    account_features = individual_df[
        individual_df["account_index"]
        == account_index
    ].copy()

    if account_features.empty:

        raise ValueError(
            "No individual explanation found "
            f"for account index {account_index}. "
            "Run individual_explanation.py first."
        )

    account_features = account_features.sort_values(
        "feature_rank"
    )

    risk_score = float(
        account_features.iloc[0][
            "risk_score"
        ]
    )

    level = risk_level(
        risk_score
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models = load_models(
        graph.x.shape[1]
    )

    # --------------------------------------------------------
    # Extract subgraph
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

    neighbors = get_neighbors(
        subset,
        local_edge_index,
        target_local_index,
    )

    local_x = local_x.to(
        next(iter(models.values())).parameters().__next__().device
    )

    local_edge_index = local_edge_index.to(
        local_x.device
    )

    baseline_score, impacts = (
        calculate_neighbor_impacts(
            models,
            local_x,
            local_edge_index,
            target_local_index,
            neighbors,
        )
    )

    impacts.sort(
        key=lambda x: x["absolute_impact"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Graph dataframe
    # --------------------------------------------------------

    graph_rows = []

    for rank, item in enumerate(
        impacts,
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

        graph_rows.append(
            {
                "neighbor_rank": rank,
                "neighbor_index": global_neighbor,
                "neighbor_id": neighbor_id,
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

    graph_df = pd.DataFrame(
        graph_rows
    )

    # --------------------------------------------------------
    # Build language
    # --------------------------------------------------------

    feature_reasons = build_feature_reasons(
        account_features
    )

    graph_reasons = build_graph_reasons(
        graph_df.head(5)
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIR
        / f"{account_id}.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Account Risk Explanation\n\n"
        )

        f.write(
            f"**Account ID:** `{account_id}`\n\n"
        )

        f.write(
            f"**Risk score:** "
            f"`{risk_score:.6f}`\n\n"
        )

        f.write(
            f"**Risk level:** `{level}`\n\n"
        )

        f.write(
            "## Model-based evidence\n\n"
        )

        for reason in feature_reasons:

            f.write(
                f"- {reason}\n"
            )

        f.write(
            "\n## Graph-based evidence\n\n"
        )

        if graph_reasons:

            for reason in graph_reasons:

                f.write(
                    f"- {reason}\n"
                )

        else:

            f.write(
                "- No direct non-self neighbors "
                "were identified in the analyzed "
                "subgraph.\n"
            )

        f.write(
            "\n## Important interpretation\n\n"
        )

        f.write(
            "These statements describe how the "
            "trained model responded to feature and "
            "neighbor perturbations. They do not "
            "establish that the account engaged in "
            "illicit activity.\n"
        )

        f.write(
            "\n## Transaction context\n\n"
        )

        for row in graph_df.head(5).itertuples():

            neighbor_index = (
                row.neighbor_index
            )

            transactions = (
                transaction_summary(
                    graph=graph,
                    subset=subset,
                    edge_mask=edge_mask,
                    account_index=account_index,
                    neighbor_index=neighbor_index,
                )
            )

            f.write(
                f"### Neighbor `{row.neighbor_id}`\n\n"
            )

            if transactions:

                for transaction in transactions:

                    f.write(
                        f"- {transaction}\n"
                    )

            else:

                f.write(
                    "- No transaction records found.\n"
                )

    print(
        f"Saved report:\n{report_path}"
    )

    return report_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate a human-readable "
            "AML model explanation."
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

    generate_report(
        account_index
    )