
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

from cp_pooling_gnn import CPPoolingGNN


SEED = 42
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 1024
NEIGHBORS = [15, 10]
HIDDEN_DIM = 64
RANK = 16
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.50

ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = ROOT / "data" / "graphs" / "training_graph.pt"
MODEL_DIR = ROOT / "data" / "graphs" / "models" / "sampled"
CHECKPOINT_PATH = MODEL_DIR / "cp_pooling_gnn_sampled_model.pt"
RESULT_PATH = (
    ROOT / "data" / "graphs" / "evaluation"
    / "phase9_cp_pooling_results.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(data, mask, shuffle):
    return NeighborLoader(
        data,
        input_nodes=mask,
        num_neighbors=NEIGHBORS,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def collect_predictions(model, loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index)
            count = batch.batch_size

            probs = torch.softmax(logits[:count], dim=1)[:, 1]
            labels = batch.y[:count]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.asarray(all_labels), np.asarray(all_probs)


def calculate_metrics(labels, probabilities, threshold=THRESHOLD):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(
            fbeta_score(labels, predictions, beta=2, zero_division=0)
        ),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "flagged": int(predictions.sum()),
        "positive_labels": int(np.sum(labels == 1)),
    }


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 9 — CP-DECOMPOSITION POOLING EXPERIMENT")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print("Loading graph...")

    payload = torch.load(
        GRAPH_PATH, map_location="cpu", weights_only=False
    )
    data = payload["data"]

    train_loader = make_loader(data, data.train_mask, True)
    val_loader = make_loader(data, data.val_mask, False)
    test_loader = make_loader(data, data.test_mask, False)

    train_labels = data.y[data.train_mask].long()
    class_counts = torch.bincount(train_labels, minlength=2).float()

    if torch.any(class_counts == 0):
        raise ValueError(f"Missing training class: {class_counts.tolist()}")

    # Inverse-frequency class weights.
    class_weights = (
        class_counts.sum() / (2.0 * class_counts)
    ).to(DEVICE)

    model = CPPoolingGNN(
        in_channels=data.x.size(1),
        hidden_channels=HIDDEN_DIM,
        out_channels=2,
        rank=RANK,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Nodes: {data.num_nodes:,}")
    print(f"Features: {data.x.size(1)}")
    print(f"Training class counts: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.cpu().tolist()}")
    print(f"Training batches per epoch: {len(train_loader)}")

    best_val_pr_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seed_total = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch.x, batch.edge_index)
            count = batch.batch_size

            seed_logits = logits[:count]
            seed_labels = batch.y[:count].long()

            loss = criterion(seed_logits, seed_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * count
            seed_total += count

        mean_train_loss = total_loss / max(seed_total, 1)

        val_y, val_probs = collect_predictions(model, val_loader)
        val_metrics = calculate_metrics(val_y, val_probs)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={mean_train_loss:.5f} | "
            f"val PR-AUC={val_metrics['pr_auc']:.5f} | "
            f"val ROC-AUC={val_metrics['roc_auc']:.5f} | "
            f"val F2@0.5={val_metrics['f2']:.5f}"
        )

        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_name": "cp_pooling_gnn",
                    "model_state_dict": model.state_dict(),
                    "input_dim": data.x.size(1),
                    "hidden_dim": HIDDEN_DIM,
                    "rank": RANK,
                    "dropout": DROPOUT,
                    "neighbors": NEIGHBORS,
                    "batch_size": BATCH_SIZE,
                    "best_epoch": best_epoch,
                    "best_val_pr_auc": best_val_pr_auc,
                    "seed": SEED,
                },
                CHECKPOINT_PATH,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation PR-AUC: {best_val_pr_auc:.6f}")
    print("Loading best checkpoint for final test evaluation...")

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    # Test data is evaluated only after model selection is complete.
    test_y, test_probs = collect_predictions(model, test_loader)
    test_metrics = calculate_metrics(test_y, test_probs)

    print("\nTEST RESULTS (threshold = 0.50)")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{key:>16}: {value:.6f}")
        else:
            print(f"{key:>16}: {value:,}")

    results = {
        "model": "CPPoolingGNN",
        "method_note": (
            "CP-style multiplicative neighbor pooling adaptation; "
            "not a full reproduction of the paper's tGNN experiments."
        ),
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_pr_auc": float(best_val_pr_auc),
        "test_metrics_at_0.50": test_metrics,
        "hyperparameters": {
            "rank": RANK,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "neighbors": NEIGHBORS,
            "batch_size": BATCH_SIZE,
        },
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nCheckpoint saved: {CHECKPOINT_PATH}")
    print(f"Results saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
