from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "training_graph.pt"
)
payload = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False,
)
data = payload["data"]
print("=" * 70)
print("TRAINING GRAPH TEST")
print("=" * 70)
print("Nodes:", data.num_nodes)
print("Edges:", data.edge_index.shape[1])
print("Features:", data.x.shape[1])
print(
    "Train nodes:",
    int(data.train_mask.sum())
)
print(
    "Validation nodes:",
    int(data.val_mask.sum())
)
print(
    "Test nodes:",
    int(data.test_mask.sum())
)
print(
    "Feature tensor shape:",
    tuple(data.x.shape)
)
print(
    "Label tensor shape:",
    tuple(data.y.shape)
)
print(
    "Edge index shape:",
    tuple(data.edge_index.shape)
)
print(
    "Training graph loaded successfully."
)