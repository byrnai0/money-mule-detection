
from pathlib import Path
from itertools import product

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = (
    ROOT / "data" / "graphs" / "evaluation"
    / "phase9_qiea" / "ensemble_predictions"
)
BASE_DIR = ROOT / "data" / "graphs" / "models" / "sampled"
QIEA_DIR = (
    ROOT / "data" / "graphs" / "models"
    / "feature_subset" / "qiea12"
)
GRAPH_FILE = ROOT / "data" / "graphs" / "training_graph.pt"

MODELS = ["sage", "gatv2", "gin"]


def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(np.int64)
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "flagged": int(pred.sum()),
    }


def load_predictions(name, split):
    path = PRED_DIR / f"{name}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    with np.load(path) as data:
        return data["labels"].copy(), data["probabilities"].copy()


def load_test_predictions(name, qiea):
    path = (
        QIEA_DIR / f"{name}_qiea12_model.pt"
        if qiea
        else BASE_DIR / f"{name}_sampled_model.pt"
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(ckpt["test_probabilities"], dtype=np.float64)


def tune_weights(validation_labels, prediction_matrix):
    # Grid-search nonnegative weights summing to 1.
    best = None
    grid = np.linspace(0, 1, 21)

    for a in grid:
        for b in grid:
            c = 1.0 - a - b
            if c < -1e-9:
                continue
            c = max(0.0, c)

            weights = np.array([a, b, c], dtype=np.float64)
            ensemble = prediction_matrix @ weights
            score = average_precision_score(validation_labels, ensemble)

            if best is None or score > best["pr_auc"]:
                best = {
                    "weights": weights,
                    "pr_auc": float(score),
                    "probabilities": ensemble,
                }

    return best


def tune_threshold_f2(y, p):
    best_threshold = 0.5
    best_f2 = -1.0

    # Threshold is selected on validation data only.
    for threshold in np.arange(0.01, 1.00, 0.01):
        score = fbeta_score(
            y, p >= threshold, beta=2, zero_division=0
        )
        if score > best_f2:
            best_f2 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f2


def report(name, y, p, threshold):
    result = metrics(y, p, threshold)
    print(f"\n{name}")
    print(f"  PR-AUC:    {result['pr_auc']:.6f}")
    print(f"  ROC-AUC:   {result['roc_auc']:.6f}")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Precision: {result['precision']:.6f}")
    print(f"  Recall:    {result['recall']:.6f}")
    print(f"  F1:        {result['f1']:.6f}")
    print(f"  F2:        {result['f2']:.6f}")
    print(f"  Flagged:   {result['flagged']:,}")


def main():
    payload = torch.load(
        GRAPH_FILE, map_location="cpu", weights_only=False
    )
    graph = payload["data"]
    y_test_graph = graph.y[graph.test_mask].cpu().numpy().astype(np.int64)

    validation_labels = None
    test_sets = {}

    for family, qiea in [("full26", False), ("qiea12", True)]:
        val_probs = []
        val_labels_ref = None
        test_probs = []

        for model in MODELS:
            name = f"{model}_qiea12" if qiea else model
            labels, probabilities = load_predictions(name, "validation")

            if val_labels_ref is None:
                val_labels_ref = labels
            elif not np.array_equal(val_labels_ref, labels):
                raise ValueError(
                    f"Validation label order mismatch for {name}"
                )

            val_probs.append(probabilities)
            test_probs.append(load_test_predictions(model, qiea))

        if validation_labels is None:
            validation_labels = val_labels_ref
        elif not np.array_equal(validation_labels, val_labels_ref):
            raise ValueError("Validation labels differ between feature sets.")

        test_probs = np.column_stack(test_probs)
        if test_probs.shape[0] != len(y_test_graph):
            raise ValueError(
                f"{family}: test probability count does not match test labels."
            )

        # QIEA checkpoints include labels; verify their order too.
        if qiea:
            for model in MODELS:
                ckpt = torch.load(
                    QIEA_DIR / f"{model}_qiea12_model.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                if not np.array_equal(
                    np.asarray(ckpt["test_labels"]), y_test_graph
                ):
                    raise ValueError(
                        f"Test label order mismatch in {model}_qiea12."
                    )

        test_sets[family] = {
            "validation_matrix": np.column_stack(val_probs),
            "test_matrix": test_probs,
        }

    if not np.array_equal(validation_labels, graph.y[graph.val_mask].cpu().numpy()):
        raise ValueError("Validation labels do not match graph mask order.")

    print("=" * 72)
    print("INDIVIDUAL MODEL VALIDATION / TEST PR-AUC")
    print("=" * 72)

    for family, qiea in [("full26", False), ("qiea12", True)]:
        val_matrix = test_sets[family]["validation_matrix"]
        test_matrix = test_sets[family]["test_matrix"]

        for i, model in enumerate(MODELS):
            val_result = average_precision_score(
                validation_labels, val_matrix[:, i]
            )
            test_result = average_precision_score(y_test_graph, test_matrix[:, i])
            print(
                f"{family:7} {model:6} "
                f"val PR-AUC={val_result:.6f} "
                f"test PR-AUC={test_result:.6f}"
            )

    print("\n" + "=" * 72)
    print("VALIDATION-TUNED ENSEMBLES; TEST EVALUATED ONCE")
    print("=" * 72)

    for family in ["full26", "qiea12"]:
        val_matrix = test_sets[family]["validation_matrix"]
        test_matrix = test_sets[family]["test_matrix"]

        best = tune_weights(validation_labels, val_matrix)
        weights = best["weights"]

        val_ensemble = val_matrix @ weights
        test_ensemble = test_matrix @ weights

        threshold, val_f2 = tune_threshold_f2(
            validation_labels, val_ensemble
        )

        print(f"\n{family.upper()}")
        print(
            "Weights (GraphSAGE, GATv2, GIN): "
            + ", ".join(f"{w:.2f}" for w in weights)
        )
        print(f"Validation PR-AUC: {best['pr_auc']:.6f}")
        print(f"Validation F2 at selected threshold: {val_f2:.6f}")

        report(
            f"{family} TEST RESULTS",
            y_test_graph,
            test_ensemble,
            threshold,
        )

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
