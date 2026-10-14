
import torch
from pathlib import Path

root = Path.cwd()
graph_file = root / "data" / "graphs" / "training_graph.pt"

bundle = torch.load(
    graph_file,
    map_location="cpu",
    weights_only=False,
)

data = bundle["data"]

print("Feature matrix:", tuple(data.x.shape))
print("Labels:", tuple(data.y.shape))
print("Train nodes:", int(data.train_mask.sum()))
print("Validation nodes:", int(data.val_mask.sum()))
print("Test nodes:", int(data.test_mask.sum()))

importance_file = (
    root / "data" / "graphs" / "evaluation"
    / "global_feature_importance.csv"
)

if importance_file.exists():
    import pandas as pd

    importance = pd.read_csv(importance_file)
    print("\nFeature names from global importance:")
    print(importance["feature"].tolist())
else:
    print("\nGlobal feature-importance CSV not found.")
