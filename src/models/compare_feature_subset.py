
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "data" / "graphs" / "models" / "sampled"
QIEA = ROOT / "data" / "graphs" / "models" / "feature_subset" / "qiea12"

models = ["sage", "gatv2", "gin"]

print(f"{'Model':<12} {'Feature set':<14} {'PR-AUC':>10} {'ROC-AUC':>10} {'F2':>10}")
print("-" * 62)

for model in models:
    paths = [
        ("26 features", BASELINE / f"{model}_sampled_model.pt"),
        ("QIEA 12", QIEA / f"{model}_qiea12_model.pt"),
    ]

    for label, path in paths:
        if not path.exists():
            print(f"{model:<12} {label:<14} CHECKPOINT MISSING")
            continue

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        metrics = checkpoint["test_metrics"]

        print(
            f"{model:<12} {label:<14} "
            f"{metrics['pr_auc']:>10.6f} "
            f"{metrics['roc_auc']:>10.6f} "
            f"{metrics['f2']:>10.6f}"
        )

    print()
