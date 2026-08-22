from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.nn import GCNConv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "training_graph.pt"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "gcn_baseline.pt"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

HIDDEN_DIM = 64
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 8
SEED = 42


# =========================================================
# Reproducibility
# =========================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# Model
# =========================================================

class GCN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels,
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        return x


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict:

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

    # These require both classes to be present.
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


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 75)
    print("PHASE 5B — GCN BASELINE")
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
    # Load data
    # -----------------------------------------------------

    print("\n[1/6] Loading training graph...")

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
    # Compute class weights using TRAIN ONLY
    # -----------------------------------------------------

    print(
        "\n[2/6] Computing class weights..."
    )

    train_labels = data.y[
        data.train_mask
    ]

    class_counts = torch.bincount(
        train_labels,
        minlength=2,
    ).float()

    total_train = class_counts.sum()

    class_weights = (
        total_train
        /
        (
            2.0
            * class_counts
        )
    )

    print(
        f"Class 0: {int(class_counts[0]):,}"
    )

    print(
        f"Class 1: {int(class_counts[1]):,}"
    )

    print(
        f"Class weights: "
        f"{class_weights.tolist()}"
    )

    # -----------------------------------------------------
    # Move graph to device
    # -----------------------------------------------------

    print(
        "\n[3/6] Moving graph to device..."
    )

    data = data.to(
        DEVICE
    )

    class_weights = class_weights.to(
        DEVICE
    )

    print("Graph moved successfully.")

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    print(
        "\n[4/6] Creating GCN..."
    )

    model = GCN(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print(model)

    # -----------------------------------------------------
    # Smoke test
    # -----------------------------------------------------

    print(
        "\n[5/6] Running GCN forward-pass smoke test..."
    )

    model.eval()

    smoke_start = time.perf_counter()

    try:

        with torch.no_grad():

            logits = model(
                data.x,
                data.edge_index,
            )

        smoke_time = (
            time.perf_counter()
            - smoke_start
        )

        print(
            f"Forward pass successful "
            f"in {smoke_time:.2f} seconds."
        )

        print(
            f"Output shape: "
            f"{tuple(logits.shape)}"
        )

    except torch.cuda.OutOfMemoryError:

        print(
            "\nCUDA OUT OF MEMORY during "
            "full-graph forward pass."
        )

        print(
            "Do not continue with full-batch training."
        )

        print(
            "We will switch this model to "
            "neighbor sampling."
        )

        raise

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print(
        "\n[6/6] Starting GCN training..."
    )

    best_val_pr_auc = -float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            data.x,
            data.edge_index,
        )

        loss = criterion(
            logits[
                data.train_mask
            ],
            data.y[
                data.train_mask
            ],
        )

        loss.backward()

        optimizer.step()

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                data.x,
                data.edge_index,
            )

            val_probabilities = torch.softmax(
                val_logits,
                dim=1,
            )[
                data.val_mask,
                1,
            ].detach().cpu().numpy()

            val_labels = data.y[
                data.val_mask
            ].detach().cpu().numpy()

        val_metrics = calculate_metrics(
            val_labels,
            val_probabilities,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.5f} | "
            f"Val PR-AUC "
            f"{val_metrics['pr_auc']:.5f} | "
            f"Val F1 "
            f"{val_metrics['f1']:.5f} | "
            f"Val Recall "
            f"{val_metrics['recall']:.5f}"
        )

        # -------------------------------------------------
        # Early stopping based on PR-AUC
        # -------------------------------------------------

        if (
            val_metrics["pr_auc"]
            > best_val_pr_auc
        ):

            best_val_pr_auc = (
                val_metrics["pr_auc"]
            )

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= PATIENCE
            ):
                print(
                    "\nEarly stopping."
                )
                break

    # -----------------------------------------------------
    # Restore best model
    # -----------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    # -----------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------

    print(
        "\nEvaluating best GCN on TEST set..."
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        test_probabilities = torch.softmax(
            logits,
            dim=1,
        )[
            data.test_mask,
            1,
        ].detach().cpu().numpy()

        test_labels = data.y[
            data.test_mask
        ].detach().cpu().numpy()

    test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
    )

    print("\n" + "=" * 75)
    print("GCN TEST RESULTS")
    print("=" * 75)

    for name, value in test_metrics.items():

        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": data.x.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "best_val_pr_auc": best_val_pr_auc,
            "test_metrics": test_metrics,
            "seed": SEED,
        },
        MODEL_FILE,
    )

    print(
        f"\nModel saved to:\n{MODEL_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()