
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]

paths = [
    ROOT / "data/graphs/models/sampled/sage_sampled_model.pt",
    ROOT / "data/graphs/models/sampled/gatv2_sampled_model.pt",
    ROOT / "data/graphs/models/sampled/gin_sampled_model.pt",
    ROOT / "data/graphs/models/feature_subset/qiea12/sage_qiea12_model.pt",
    ROOT / "data/graphs/models/feature_subset/qiea12/gatv2_qiea12_model.pt",
    ROOT / "data/graphs/models/feature_subset/qiea12/gin_qiea12_model.pt",
]

for path in paths:
    print(f"\n{path.name}")
    if not path.exists():
        print("  MISSING")
        continue

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    print("  Keys:", sorted(checkpoint.keys()))
    for key in ("val_labels", "val_probabilities", "test_labels", "test_probabilities"):
        if key in checkpoint:
            print(f"  {key}: {len(checkpoint[key])} entries")
        else:
            print(f"  {key}: NOT SAVED")
