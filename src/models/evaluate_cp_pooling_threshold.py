
from pathlib import Path
import json

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

from cp_pooling_gnn import CPPoolingGNN

ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = ROOT / "data" / "graphs" / "training_graph.pt"
CHECKPOINT_PATH = (
    ROOT / "data" / "graphs" / "models" / "sampled"
    / "cp_pooling_gnn_sampled_model.pt"
)
RESULT_PATH = (
    ROOT / "data" / "graphs" / "evaluation"
    / "phase9_cp_pooling_threshold.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, loader):
    model.eval()
    labels_all, probs_all = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index)
            n = batch.batch_size

            probs_all.extend(
                torch.softmax(logits[:n], dim=1)[:, 1].cpu().numpy()
            )
            labels_all.extend(batch.y[:n].cpu().numpy())

    return np.asarray(labels_all), np.asarray(probs_all)


def metrics(y, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probs)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "flagged": int(pred.sum()),
    }


def main():
    print(f"Device: {DEVICE}")

    graph_payload = torch.load(
        GRAPH_PATH, map_location="cpu", weights_only=False
    )
    data = graph_payload["data"]

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )

    model = CPPoolingGNN(
        in_channels=checkpoint["input_dim"],
        hidden_channels=checkpoint["hidden_dim"],
        out_channels=2,
        rank=checkpoint["rank"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    neighbors = checkpoint["neighbors"]
    batch_size = checkpoint["batch_size"]

    def loader(mask):
        return NeighborLoader(
            data,
            input_nodes=mask,
            num_neighbors=neighbors,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    print("Collecting validation predictions...")
    val_y, val_probs = predict(model, loader(data.val_mask))

    print("Collecting test predictions...")
    test_y, test_probs = predict(model, loader(data.test_mask))

    # Choose the threshold using validation F2 only.
    best = None
    for threshold in np.linspace(0.01, 0.99, 991):
        result = metrics(val_y, val_probs, threshold)
        if best is None or (
            result["f2"], result["precision"]
        ) > (best["f2"], best["precision"]):
            best = result

    test_result = metrics(test_y, test_probs, best["threshold"])

    print("\nVALIDATION: best threshold by F2")
    for key, value in best.items():
        print(f"{key:>12}: {value:.6f}")

    print("\nTEST: at validation-selected threshold")
    for key, value in test_result.items():
        if isinstance(value, float):
            print(f"{key:>12}: {value:.6f}")
        else:
            print(f"{key:>12}: {value:,}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": "CPPoolingGNN",
            "threshold_selection": "validation F2",
            "validation_best": best,
            "test_at_validation_threshold": test_result,
        }, f, indent=2)

    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
