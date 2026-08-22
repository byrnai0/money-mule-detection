from __future__ import annotations

from pathlib import Path
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 42

HIDDEN_DIM = 64
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 20
PATIENCE = 5

BATCH_SIZE = 1024

# Number of neighbors sampled at each GNN layer.
# Two layers -> two neighborhood hops.
NUM_NEIGHBORS = [15, 10]


def set_seed():

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    metrics = {
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f2": fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true)) == 2:

        metrics["pr_auc"] = (
            average_precision_score(
                y_true,
                probabilities,
            )
        )

        metrics["roc_auc"] = (
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    else:

        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")

    return metrics


def main(model_name: str):

    set_seed()

    print("=" * 75)
    print(
        f"PHASE 5C — SAMPLED {model_name.upper()} EXPERIMENT"
    )
    print("=" * 75)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------
    # Load graph
    # -----------------------------------------------------

    print("\n[1/8] Loading training graph...")

    payload = torch.load(
        DATA_FILE,
        map_location="cpu",
        weights_only=False,
    )

    data = payload["data"]

    print(
        f"Nodes: {data.num_nodes:,}"
    )

    print(
        f"Edges: {data.edge_index.shape[1]:,}"
    )

    print(
        f"Features: {data.x.shape[1]}"
    )

    # -----------------------------------------------------
    # Class weights
    # -----------------------------------------------------

    print(
        "\n[2/8] Computing class weights..."
    )

    train_labels = data.y[
        data.train_mask
    ]

    class_counts = torch.bincount(
        train_labels,
        minlength=2,
    ).float()

    total = class_counts.sum()

    class_weights = (
        total
        /
        (2 * class_counts)
    )

    print(
        f"Class counts: "
        f"{class_counts.tolist()}"
    )

    print(
        f"Class weights: "
        f"{class_weights.tolist()}"
    )

    # -----------------------------------------------------
    # Training loader
    # -----------------------------------------------------

    print(
        "\n[3/8] Creating neighbor-sampled loader..."
    )

    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=data.train_mask,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # -----------------------------------------------------
    # Validation loader
    # -----------------------------------------------------

    print(
        "\n[4/8] Creating validation loader..."
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=data.val_mask,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # -----------------------------------------------------
    # Test loader
    # -----------------------------------------------------

    print(
        "\n[5/8] Creating test loader..."
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=data.test_mask,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    print(
        "\n[6/8] Building model..."
    )

    model = build_model(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    class_weights = class_weights.to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print(
        "\n[7/8] Training..."
    )

    best_pr_auc = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        batches = 0

        start = time.perf_counter()

        for batch in train_loader:

            batch = batch.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch.x,
                batch.edge_index,
            )

            # In NeighborLoader, the first
            # batch.batch_size nodes correspond
            # to the seed/input nodes.
            logits_seed = logits[
                :batch.batch_size
            ]

            labels_seed = batch.y[
                :batch.batch_size
            ]

            loss = criterion(
                logits_seed,
                labels_seed,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            batches += 1

        epoch_loss = (
            total_loss
            / max(batches, 1)
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        val_probabilities = []
        val_labels = []

        with torch.no_grad():

            for batch in val_loader:

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

                probabilities = torch.softmax(
                    seed_logits,
                    dim=1,
                )[:, 1]

                val_probabilities.extend(
                    probabilities.cpu().numpy()
                )

                val_labels.extend(
                    batch.y[
                        :batch.batch_size
                    ].cpu().numpy()
                )

        val_probabilities = np.asarray(
            val_probabilities
        )

        val_labels = np.asarray(
            val_labels
        )

        val_metrics = calculate_metrics(
            val_labels,
            val_probabilities,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {epoch_loss:.5f} | "
            f"PR-AUC {val_metrics['pr_auc']:.5f} | "
            f"F1 {val_metrics['f1']:.5f} | "
            f"Recall {val_metrics['recall']:.5f} | "
            f"{elapsed:.1f}s"
        )

        if (
            val_metrics["pr_auc"]
            > best_pr_auc
        ):

            best_pr_auc = (
                val_metrics["pr_auc"]
            )

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping."
                )

                break

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint produced."
        )

    model.load_state_dict(
        best_state
    )

    # -----------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------

    print(
        "\n[8/8] Evaluating test set..."
    )

    model.eval()

    test_probabilities = []
    test_labels = []

    with torch.no_grad():

        for batch in test_loader:

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

            probabilities = torch.softmax(
                seed_logits,
                dim=1,
            )[:, 1]

            test_probabilities.extend(
                probabilities.cpu().numpy()
            )

            test_labels.extend(
                batch.y[
                    :batch.batch_size
                ].cpu().numpy()
            )

    test_probabilities = np.asarray(
        test_probabilities
    )

    test_labels = np.asarray(
        test_labels
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
    )

    print("\n" + "=" * 75)
    print(
        f"SAMPLED {model_name.upper()} TEST RESULTS"
    )
    print("=" * 75)

    for name, value in test_metrics.items():

        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        MODEL_DIR
        / f"{model_name}_sampled_model.pt"
    )

    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "input_dim": data.x.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "neighbors": NUM_NEIGHBORS,
            "batch_size": BATCH_SIZE,
            "test_metrics": test_metrics,
            "test_probabilities": test_probabilities,
            "seed": SEED,
        },
        output_file,
    )

    print(
        f"\nSaved:\n{output_file}"
    )

    print("=" * 75)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "gcn",
            "gat",
            "gatv2",
            "sage",
            "gin",
            "cheb",
        ],
    )

    args = parser.parse_args()

    main(
        args.model
    )