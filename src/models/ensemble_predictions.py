from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader

from architectures import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
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

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_labels.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "ensemble_predictions.csv"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]

MODELS = [
    "sage",
    "gatv2",
    "gin",
]


def get_loader(data, mask, shuffle=False):

    return NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=mask,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def load_model(
    model_name: str,
    input_dim: int,
):

    model_file = (
        MODEL_DIR
        / f"{model_name}_sampled_model.pt"
    )

    payload = torch.load(
        model_file,
        map_location="cpu",
        weights_only=False,
    )

    model = build_model(
        name=model_name,
        in_channels=input_dim,
        hidden_channels=64,
        dropout=0.30,
    )

    model.load_state_dict(
        payload["model_state_dict"]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


def predict(
    model,
    loader,
):

    probabilities = []
    node_ids = []

    with torch.no_grad():

        for batch in loader:

            seed_nodes = batch.n_id[
                :batch.batch_size
            ]

            batch = batch.to(
                DEVICE
            )

            logits = model(
                batch.x,
                batch.edge_index,
            )

            seed_logits = logits[
                :batch.batch_size
            ]

            probs = torch.softmax(
                seed_logits,
                dim=1,
            )[:, 1]

            probabilities.extend(
                probs.cpu().numpy()
            )

            node_ids.extend(
                seed_nodes.numpy()
            )

    return (
        np.asarray(node_ids),
        np.asarray(probabilities),
    )


def main():

    print("=" * 75)
    print("PHASE 6A — ENSEMBLE PREDICTION EXTRACTION")
    print("=" * 75)

    # ---------------------------------------------------------
    # Load graph
    # ---------------------------------------------------------

    print("\n[1/5] Loading training graph...")

    payload = torch.load(
        DATA_FILE,
        map_location="cpu",
        weights_only=False,
    )

    data = payload["data"]
    account_list = payload["account_list"]

    print(
        f"Nodes: {data.num_nodes:,}"
    )

    # ---------------------------------------------------------
    # Create loaders
    # ---------------------------------------------------------

    print(
        "\n[2/5] Creating validation/test loaders..."
    )

    val_loader = get_loader(
        data,
        data.val_mask,
        shuffle=False,
    )

    test_loader = get_loader(
        data,
        data.test_mask,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Base dataframe
    # ---------------------------------------------------------

    results = pd.DataFrame(
        {
            "node_id": np.arange(
                data.num_nodes
            ),
            "account_id": account_list,
            "is_illicit": (
                data.y.numpy()
            ),
        }
    )

    # ---------------------------------------------------------
    # Extract predictions
    # ---------------------------------------------------------

    print(
        "\n[3/5] Extracting model predictions..."
    )

    for model_name in MODELS:

        print(
            f"\n  → {model_name.upper()}"
        )

        model = load_model(
            model_name,
            data.x.shape[1],
        )

        val_nodes, val_probs = predict(
            model,
            val_loader,
        )

        test_nodes, test_probs = predict(
            model,
            test_loader,
        )

        val_column = (
            f"{model_name}_val_prob"
        )

        test_column = (
            f"{model_name}_test_prob"
        )

        results[val_column] = np.nan
        results[test_column] = np.nan

        results.loc[
            val_nodes,
            val_column,
        ] = val_probs

        results.loc[
            test_nodes,
            test_column,
        ] = test_probs

        print(
            f"     validation predictions: "
            f"{len(val_probs):,}"
        )

        print(
            f"     test predictions: "
            f"{len(test_probs):,}"
        )

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    print(
        "\n[4/5] Running prediction alignment checks..."
    )

    for model_name in MODELS:

        val_column = (
            f"{model_name}_val_prob"
        )

        test_column = (
            f"{model_name}_test_prob"
        )

        val_count = (
            results[val_column]
            .notna()
            .sum()
        )

        test_count = (
            results[test_column]
            .notna()
            .sum()
        )

        print(
            f"{model_name.upper():<8} "
            f"val={val_count:,} "
            f"test={test_count:,}"
        )

        expected_val = int(
            data.val_mask.sum()
        )

        expected_test = int(
            data.test_mask.sum()
        )

        if val_count != expected_val:
            raise RuntimeError(
                f"{model_name}: validation "
                "prediction count mismatch."
            )

        if test_count != expected_test:
            raise RuntimeError(
                f"{model_name}: test "
                "prediction count mismatch."
            )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print(
        "\n[5/5] Saving predictions..."
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 75)
    print("PHASE 6.1 COMPLETE")
    print("=" * 75)

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print(
        "\nColumns:"
    )

    for column in results.columns:
        print(f"  - {column}")

    print("=" * 75)


if __name__ == "__main__":
    main()