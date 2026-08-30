from __future__ import annotations

from pathlib import Path
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

from architectures import GINModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
    / "temporal_experiment.pt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
    / "temporal_gin_model.pt"
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
NUM_NEIGHBORS = [15, 10]


def set_seed():

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def evaluate_predictions(
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


def make_loader(
    data,
    input_nodes,
    shuffle: bool,
):

    return NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=input_nodes,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def predict(
    model,
    data,
    loader,
):

    model.eval()

    probabilities = []

    with torch.no_grad():

        for batch in loader:

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

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )


def main():

    set_seed()

    start = time.perf_counter()

    print("=" * 75)
    print("Phase 7.6.2 - Temporal Ensemble")
    print("=" * 75)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # Load temporal experiment
    # ---------------------------------------------------------

    print(
        "\n[1/7] Loading temporal experiment..."
    )

    payload = torch.load(
        DATA_FILE,
        map_location="cpu",
        weights_only=False,
    )

    train_data = payload["train"]
    val_data = payload["validation"]
    test_data = payload["test"]

    print(
        f"Train nodes: "
        f"{train_data.num_nodes:,}"
    )

    print(
        f"Validation nodes: "
        f"{val_data.num_nodes:,}"
    )

    print(
        f"Test nodes: "
        f"{test_data.num_nodes:,}"
    )

    # ---------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------

    print(
        "\n[2/7] Computing train class weights..."
    )

    train_labels = train_data.y

    class_counts = torch.bincount(
        train_labels,
        minlength=2,
    ).float()

    total = class_counts.sum()

    class_weights = (
        total
        /
        (2.0 * class_counts)
    )

    print(
        f"Class 0: {int(class_counts[0]):,}"
    )

    print(
        f"Class 1: {int(class_counts[1]):,}"
    )

    print(
        f"Weights: "
        f"{class_weights.tolist()}"
    )

    # ---------------------------------------------------------
    # Training loader
    # ---------------------------------------------------------

    print(
        "\n[3/7] Creating training sampler..."
    )

    train_loader = make_loader(
        train_data,
        torch.arange(
            train_data.num_nodes
        ),
        shuffle=True,
    )

    # ---------------------------------------------------------
    # Validation/test loaders
    # ---------------------------------------------------------

    print(
        "\n[4/7] Creating validation/test samplers..."
    )

    val_loader = make_loader(
        val_data,
        torch.arange(
            val_data.num_nodes
        ),
        shuffle=False,
    )

    test_loader = make_loader(
        test_data,
        torch.arange(
            test_data.num_nodes
        ),
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------

    print(
        "\n[5/7] Building GIN..."
    )

    model = GINModel(
        in_channels=train_data.x.shape[1],
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
        weight=class_weights.to(
            DEVICE
        )
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print(
        "\n[6/7] Training on TRAIN period..."
    )

    best_val_pr_auc = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        batches = 0

        epoch_start = time.perf_counter()

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

            seed_logits = logits[
                :batch.batch_size
            ]

            seed_labels = batch.y[
                :batch.batch_size
            ]

            loss = criterion(
                seed_logits,
                seed_labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        average_loss = (
            total_loss
            / max(
                batches,
                1,
            )
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        val_probabilities = predict(
            model,
            val_data,
            val_loader,
        )

        val_labels = (
            val_data.y
            .numpy()
        )

        val_metrics = (
            evaluate_predictions(
                val_labels,
                val_probabilities,
            )
        )

        elapsed = (
            time.perf_counter()
            - epoch_start
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {average_loss:.5f} | "
            f"Val PR-AUC "
            f"{val_metrics['pr_auc']:.5f} | "
            f"Val F1 "
            f"{val_metrics['f1']:.5f} | "
            f"Val Recall "
            f"{val_metrics['recall']:.5f} | "
            f"{elapsed:.1f}s"
        )

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
            "No valid temporal checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    # ---------------------------------------------------------
    # Final validation / test evaluation
    # ---------------------------------------------------------

    print(
        "\n[7/7] Evaluating frozen temporal model..."
    )

    val_probabilities = predict(
        model,
        val_data,
        val_loader,
    )

    test_probabilities = predict(
        model,
        test_data,
        test_loader,
    )

    val_labels = (
        val_data.y
        .numpy()
    )

    test_labels = (
        test_data.y
        .numpy()
    )

    val_metrics = evaluate_predictions(
        val_labels,
        val_probabilities,
    )

    test_metrics = evaluate_predictions(
        test_labels,
        test_probabilities,
    )

    print("\n" + "=" * 75)
    print("TEMPORAL VALIDATION RESULTS")
    print("=" * 75)

    for name, value in val_metrics.items():

        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    print("\n" + "=" * 75)
    print("TEMPORAL TEST RESULTS")
    print("=" * 75)

    for name, value in test_metrics.items():

        print(
            f"{name.upper():<12}: "
            f"{value:.6f}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_name": "GraphSAGE",
            "model_state_dict": model.state_dict(),
            "input_dim": train_data.x.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "neighbors": NUM_NEIGHBORS,
            "batch_size": BATCH_SIZE,
            "best_val_pr_auc": best_val_pr_auc,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "validation_probabilities": val_probabilities,
            "test_probabilities": test_probabilities,
            "seed": SEED,
            "train_cutoff": payload["train_cutoff"],
            "test_cutoff": payload["test_cutoff"],
        },
        OUTPUT_FILE,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"\nTemporal model saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Runtime: {elapsed:.2f}s"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()