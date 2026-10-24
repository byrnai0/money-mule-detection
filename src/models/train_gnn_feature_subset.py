
from __future__ import annotations

from pathlib import Path
import argparse
import copy
import json
import random
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

DATA_FILE = PROJECT_ROOT / "data" / "graphs" / "training_graph.pt"

# Separate folder: original 26-feature checkpoints remain untouched.
MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "models"
    / "feature_subset"
    / "qiea12"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
HIDDEN_DIM = 64
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]

# Indices must match feature_manifest.csv column order.
FEATURE_INDICES = [
    0, 2, 7, 9, 13, 15, 16, 20, 22, 23, 24, 25
]

FEATURE_NAMES = [
    "log1p_in_transaction_count",
    "log1p_unique_senders",
    "log1p_incoming_active_days",
    "log1p_active_days",
    "log1p_incoming_payment_format_diversity",
    "log1p_receiving_currency_diversity",
    "log1p_payment_currency_diversity",
    "incoming_currency_abs_z_mean",
    "outgoing_log_amount_mean",
    "outgoing_log_amount_std",
    "outgoing_log_amount_max",
    "outgoing_currency_abs_z_max",
]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(np.int64)

    metrics = {
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(
            fbeta_score(y_true, predictions, beta=2, zero_division=0)
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["pr_auc"] = float(
            average_precision_score(y_true, probabilities)
        )
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )
    else:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")

    return metrics


def evaluate(model, loader):
    model.eval()
    probabilities_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)

            logits = model(batch.x, batch.edge_index)
            seed_logits = logits[:batch.batch_size]

            probabilities = torch.softmax(seed_logits, dim=1)[:, 1]

            probabilities_all.extend(
                probabilities.cpu().numpy().tolist()
            )
            labels_all.extend(
                batch.y[:batch.batch_size].cpu().numpy().tolist()
            )

    return (
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(probabilities_all, dtype=np.float32),
    )


def main(model_name: str):
    set_seed()

    print("=" * 72)
    print(f"QIEA 12-FEATURE SAMPLED GNN: {model_name.upper()}")
    print("=" * 72)
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1/6] Loading graph...")
    payload = torch.load(
        DATA_FILE,
        map_location="cpu",
        weights_only=False,
    )
    data = payload["data"]

    original_feature_count = data.x.shape[1]

    if original_feature_count != 26:
        raise ValueError(
            f"Expected 26 original features, got {original_feature_count}. "
            "Check the graph bundle and feature manifest."
        )

    if max(FEATURE_INDICES) >= original_feature_count:
        raise ValueError("A selected feature index is out of range.")

    # Keep only the QIEA-selected feature columns.
    data.x = data.x[:, FEATURE_INDICES].contiguous()

    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.edge_index.shape[1]:,}")
    print(f"Original features: {original_feature_count}")
    print(f"Selected features: {data.x.shape[1]}")

    for index, name in zip(FEATURE_INDICES, FEATURE_NAMES):
        print(f"  [{index:02d}] {name}")

    print("\n[2/6] Calculating class weights...")
    train_labels = data.y[data.train_mask]
    class_counts = torch.bincount(train_labels, minlength=2).float()

    if torch.any(class_counts == 0):
        raise ValueError(
            f"Training split must contain both classes: {class_counts.tolist()}"
        )

    class_weights = class_counts.sum() / (2 * class_counts)
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")

    print("\n[3/6] Creating neighbor loaders...")
    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=data.train_mask,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=data.val_mask,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=data.test_mask,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("\n[4/6] Building model...")
    model = build_model(
        name=model_name,
        in_channels=len(FEATURE_INDICES),
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    print("\n[5/6] Training...")
    best_pr_auc = -float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        start = time.perf_counter()

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch.x, batch.edge_index)
            seed_logits = logits[:batch.batch_size]
            seed_labels = batch.y[:batch.batch_size]

            loss = criterion(seed_logits, seed_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        epoch_loss = total_loss / max(batches, 1)

        val_labels, val_probs = evaluate(model, val_loader)
        val_metrics = calculate_metrics(val_labels, val_probs)
        # Save validation predictions from the current best checkpoint.
        # These are retained when this epoch becomes the best checkpoint.

        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        elapsed = time.perf_counter() - start
        print(
            f"Epoch {epoch:02d} | "
            f"Loss {epoch_loss:.5f} | "
            f"Val PR-AUC {val_metrics['pr_auc']:.5f} | "
            f"Val F1 {val_metrics['f1']:.5f} | "
            f"Val Recall {val_metrics['recall']:.5f} | "
            f"{elapsed:.1f}s"
        )

        
        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            # Save validation predictions for this best checkpoint.
            best_val_labels = val_labels.copy()
            best_val_probabilities = val_probs.copy()

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    # Evaluate the test set only after selecting the best validation checkpoint.
    print("\n[6/6] Evaluating best checkpoint on test set...")
    test_labels, test_probs = evaluate(model, test_loader)
    test_metrics = calculate_metrics(test_labels, test_probs)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation PR-AUC: {best_pr_auc:.6f}")
    print("Test metrics at threshold 0.50:")
    for name, value in test_metrics.items():
        print(f"  {name.upper():<12}: {value:.6f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = MODEL_DIR / f"{model_name}_qiea12_model.pt"
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": best_state,
            "input_dim": len(FEATURE_INDICES),
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "neighbors": NUM_NEIGHBORS,
            "batch_size": BATCH_SIZE,
            "feature_indices": FEATURE_INDICES,
            "feature_names": FEATURE_NAMES,
            "best_epoch": best_epoch,
            "best_val_pr_auc": best_pr_auc,
            "test_metrics": test_metrics,
            "test_probabilities": test_probs,
            "test_labels": test_labels,
            "seed": SEED,
            "val_labels": best_val_labels,
            "val_probabilities": best_val_probabilities,
        },
        checkpoint_path,
    )

    with open(MODEL_DIR / f"{model_name}_qiea12_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved history: {MODEL_DIR / f'{model_name}_qiea12_history.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a sampled GNN using the QIEA-selected 12 features."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["gcn", "gat", "gatv2", "sage", "gin", "cheb"],
    )
    args = parser.parse_args()
    main(args.model)
