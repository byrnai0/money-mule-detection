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

EPOCHS = 30
PATIENCE = 7


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


def evaluate(
    model,
    data,
    mask,
):

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[mask, 1]

    y_true = data.y[
        mask
    ].detach().cpu().numpy()

    probabilities = probabilities.detach().cpu().numpy()

    return calculate_metrics(
        y_true,
        probabilities,
    ), probabilities


def main(model_name: str):

    set_seed()

    print("=" * 75)
    print(
        f"PHASE 5C — {model_name.upper()} EXPERIMENT"
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
    # Class weights
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
    # Move to GPU
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

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    print(
        "\n[4/6] Building model..."
    )

    model = build_model(
        name=model_name,
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(
        DEVICE
    )

    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -----------------------------------------------------
    # Smoke test
    # -----------------------------------------------------

    print(
        "\n[5/6] Full-graph forward-pass test..."
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
            f"in {smoke_time:.2f}s"
        )

        print(
            f"Output shape: "
            f"{tuple(logits.shape)}"
        )

    except torch.cuda.OutOfMemoryError:

        print(
            "\nCUDA OUT OF MEMORY."
        )

        print(
            "This architecture needs "
            "neighbor-sampled training."
        )

        raise

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print(
        "\n[6/6] Training..."
    )

    best_pr_auc = -float("inf")
    best_state = None
    patience_counter = 0

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
            logits[data.train_mask],
            data.y[data.train_mask],
        )

        loss.backward()

        optimizer.step()

        val_metrics, _ = evaluate(
            model,
            data,
            data.val_mask,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.5f} | "
            f"PR-AUC {val_metrics['pr_auc']:.5f} | "
            f"F1 {val_metrics['f1']:.5f} | "
            f"Recall {val_metrics['recall']:.5f}"
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

    # -----------------------------------------------------
    # Restore best
    # -----------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint created."
        )

    model.load_state_dict(
        best_state
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    print(
        "\nEvaluating test set..."
    )

    test_metrics, test_probabilities = evaluate(
        model,
        data,
        data.test_mask,
    )

    print("\n" + "=" * 75)
    print(
        f"{model_name.upper()} TEST RESULTS"
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
        / f"{model_name}_model.pt"
    )

    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "input_dim": data.x.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "best_val_pr_auc": best_pr_auc,
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