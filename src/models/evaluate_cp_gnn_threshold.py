
from pathlib import Path
import sys

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

# Allow importing cp_gnn.py from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cp_gnn import CPGNNModel


ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = ROOT / "data" / "graphs" / "training_graph.pt"
CHECKPOINT_PATH = (
    ROOT / "data" / "graphs" / "models" / "sampled"
    / "cp_gnn_sampled_model.pt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_predictions(model, loader, device):
    probabilities = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)

            # Only the first batch.batch_size nodes are the target seed nodes.
            seed_count = batch.batch_size
            probs = torch.softmax(logits[:seed_count], dim=1)[:, 1]

            probabilities.extend(probs.cpu().numpy())
            labels.extend(batch.y[:seed_count].cpu().numpy())

    return np.asarray(labels), np.asarray(probabilities)


def metrics_at_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "f2": fbeta_score(
            y_true, predictions, beta=2, zero_division=0
        ),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "flagged": int(predictions.sum()),
    }


def main():
    print(f"Device: {DEVICE}")
    print("Loading graph and checkpoint...")

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    graph_payload = torch.load(
        GRAPH_PATH, map_location="cpu", weights_only=False
    )
    data = graph_payload["data"]

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )

    model = CPGNNModel(
        in_channels=checkpoint["input_dim"],
        hidden_channels=checkpoint["hidden_dim"],
        out_channels=2,
        rank=16,
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    neighbors = checkpoint.get("neighbors", [15, 10])
    batch_size = checkpoint.get("batch_size", 1024)

    print(f"Graph nodes: {data.num_nodes:,}")
    print(f"Validation nodes: {int(data.val_mask.sum()):,}")
    print(f"Test nodes: {int(data.test_mask.sum()):,}")
    print(f"Neighbor sampling: {neighbors}")
    print("\nCollecting validation predictions...")

    val_loader = NeighborLoader(
        data,
        input_nodes=data.val_mask,
        num_neighbors=neighbors,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    val_y, val_probs = collect_predictions(
        model, val_loader, DEVICE
    )

    print("Collecting test predictions...")
    test_loader = NeighborLoader(
        data,
        input_nodes=data.test_mask,
        num_neighbors=neighbors,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_y, test_probs = collect_predictions(
        model, test_loader, DEVICE
    )

    # Choose threshold using validation F2 only.
    # Test labels are not used to select the threshold.
    thresholds = np.linspace(0.01, 0.99, 991)
    candidates = []

    for threshold in thresholds:
        result = metrics_at_threshold(val_y, val_probs, threshold)
        candidates.append(result)

    # Maximize F2; if tied, prefer higher precision.
    best = max(
        candidates,
        key=lambda item: (item["f2"], item["precision"]),
    )
    best_threshold = best["threshold"]

    test_result = metrics_at_threshold(
        test_y, test_probs, best_threshold
    )

    print("\n" + "=" * 55)
    print("CP-GNN: VALIDATION-SELECTED THRESHOLD")
    print("=" * 55)
    print(f"Selected threshold: {best_threshold:.3f}")
    print(f"Validation precision: {best['precision']:.6f}")
    print(f"Validation recall:    {best['recall']:.6f}")
    print(f"Validation F1:        {best['f1']:.6f}")
    print(f"Validation F2:        {best['f2']:.6f}")

    print("\n" + "=" * 55)
    print("CP-GNN: TEST RESULTS AT THAT THRESHOLD")
    print("=" * 55)
    for name, value in test_result.items():
        if isinstance(value, float):
            print(f"{name:>12}: {value:.6f}")
        else:
            print(f"{name:>12}: {value:,}")

    output_path = (
        ROOT / "data" / "graphs" / "evaluation"
        / "phase9_cp_gnn_threshold.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    results = {
        "model": "CP-GNN",
        "threshold_selection_metric": "validation F2",
        "validation_best": best,
        "test_at_validation_threshold": test_result,
        "validation_node_count": int(len(val_y)),
        "test_node_count": int(len(test_y)),
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
