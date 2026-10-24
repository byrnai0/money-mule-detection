
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

from architectures import build_model

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "graphs" / "training_graph.pt"

BASELINE_DIR = ROOT / "data" / "graphs" / "models" / "sampled"
QIEA_DIR = (
    ROOT / "data" / "graphs" / "models"
    / "feature_subset" / "qiea12"
)
OUTPUT_DIR = ROOT / "data" / "graphs" / "evaluation" / "phase9_qiea" / "ensemble_predictions"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]

MODELS = [
    ("sage", BASELINE_DIR / "sage_sampled_model.pt", False),
    ("gatv2", BASELINE_DIR / "gatv2_sampled_model.pt", False),
    ("gin", BASELINE_DIR / "gin_sampled_model.pt", False),
    ("sage_qiea12", QIEA_DIR / "sage_qiea12_model.pt", True),
    ("gatv2_qiea12", QIEA_DIR / "gatv2_qiea12_model.pt", True),
    ("gin_qiea12", QIEA_DIR / "gin_qiea12_model.pt", True),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = torch.load(
        DATA_FILE, map_location="cpu", weights_only=False
    )
    original_data = payload["data"]

    for name, checkpoint_path, use_qiea_features in MODELS:
        print(f"\nProcessing {name}...")

        if not checkpoint_path.exists():
            print(f"  Missing checkpoint: {checkpoint_path}")
            continue

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        # Use the same selected columns that were used during QIEA training.
        data = original_data.clone()
        if use_qiea_features:
            indices = checkpoint["feature_indices"]
            data.x = data.x[:, indices].contiguous()

        if data.x.shape[1] != checkpoint["input_dim"]:
            raise ValueError(
                f"{name}: input dimension mismatch: "
                f"graph has {data.x.shape[1]}, "
                f"checkpoint expects {checkpoint['input_dim']}"
            )

        model = build_model(
            name=checkpoint["model_name"],
            in_channels=checkpoint["input_dim"],
            hidden_channels=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
        ).to(DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        loader = NeighborLoader(
            data,
            num_neighbors=checkpoint.get("neighbors", NUM_NEIGHBORS),
            input_nodes=data.val_mask,
            batch_size=checkpoint.get("batch_size", BATCH_SIZE),
            shuffle=False,
            num_workers=0,
        )

        probabilities = []
        labels = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index)
                seed_logits = logits[:batch.batch_size]
                probs = torch.softmax(seed_logits, dim=1)[:, 1]

                probabilities.extend(probs.cpu().numpy().tolist())
                labels.extend(
                    batch.y[:batch.batch_size].cpu().numpy().tolist()
                )

        probabilities = np.asarray(probabilities, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        output_path = OUTPUT_DIR / f"{name}_validation.npz"
        np.savez_compressed(
            output_path,
            probabilities=probabilities,
            labels=labels,
        )

        print(f"  Validation nodes: {len(labels):,}")
        print(f"  Saved: {output_path}")

    print("\nRecovery complete.")


if __name__ == "__main__":
    main()
