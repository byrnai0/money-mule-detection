
from pathlib import Path
import itertools

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

# Columns: full-feature SAGE, GATv2, GIN, then QIEA-12 SAGE, GATv2, GIN.
MODEL_NAMES = [
    "sage", "gatv2", "gin",
    "sage_qiea12", "gatv2_qiea12", "gin_qiea12",
]


def load_validation(name):
    path = PRED_DIR / f"{name}_validation.npz"
    with np.load(path) as d:
        return d["labels"].copy(), d["probabilities"].copy()


def load_test(name):
    if name.endswith("_qiea12"):
        path = QIEA_DIR / f"{name}_model.pt"
    else:
        path = BASE_DIR / f"{name}_sampled_model.pt"

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(checkpoint["test_probabilities"], dtype=np.float64)


def get_metrics(y, p, threshold):
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


def tune_threshold(y, p):
    thresholds = np.arange(0.01, 1.00, 0.01)
    scores = [
        fbeta_score(y, p >= t, beta=2, zero_division=0)
        for t in thresholds
    ]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def main():
    graph = torch.load(
        GRAPH_FILE, map_location="cpu", weights_only=False
    )["data"]
    y_val_expected = graph.y[graph.val_mask].cpu().numpy().astype(np.int64)
    y_test = graph.y[graph.test_mask].cpu().numpy().astype(np.int64)

    val_columns = []
    y_val = None

    for name in MODEL_NAMES:
        labels, probabilities = load_validation(name)
        if y_val is None:
            y_val = labels
        elif not np.array_equal(y_val, labels):
            raise ValueError(f"Validation label order mismatch: {name}")
        val_columns.append(probabilities)

    if not np.array_equal(y_val, y_val_expected):
        raise ValueError("Validation predictions do not match graph mask order.")

    val_matrix = np.column_stack(val_columns)
    test_matrix = np.column_stack([load_test(name) for name in MODEL_NAMES])

    if len(y_test) != test_matrix.shape[0]:
        raise ValueError("Test probability count does not match test labels.")

    # Confirm saved QIEA test labels match the graph's test order.
    for name in MODEL_NAMES[3:]:
        ckpt = torch.load(
            QIEA_DIR / f"{name}_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not np.array_equal(np.asarray(ckpt["test_labels"]), y_test):
            raise ValueError(f"Test label order mismatch: {name}")

    # Full-feature baseline: the previously selected 0.85 / 0.15 / 0.
    baseline_weights = np.array([0.85, 0.15, 0, 0, 0, 0])
    baseline_val = val_matrix @ baseline_weights
    baseline_test = test_matrix @ baseline_weights
    baseline_threshold, _ = tune_threshold(y_val, baseline_val)

    print("FULL-FEATURE BASELINE")
    print("Weights:", baseline_weights.tolist())
    print("Validation PR-AUC:", round(
        average_precision_score(y_val, baseline_val), 6
    ))
    print("Validation-selected threshold:", baseline_threshold)

    baseline_result = get_metrics(y_test, baseline_test, baseline_threshold)
    for key, value in baseline_result.items():
        print(f"Test {key}: {value:.6f}" if isinstance(value, float)
              else f"Test {key}: {value}")

    
    # Efficient hybrid search:
    # Precompute a small set of full-feature and QIEA group ensembles.
    # This avoids recalculating PR-AUC for thousands of nested combinations.

    best = None
    group_grid = np.arange(0.0, 1.0001, 0.10)
    qiea_shares = np.arange(0.0, 0.3001, 0.05)

    full_candidates = []
    qiea_candidates = []

    for a in group_grid:
        for b in group_grid:
            c = 1.0 - a - b
            if c < -1e-9:
                continue

            w = np.array([a, b, max(0.0, c)], dtype=np.float64)
            w /= w.sum()
            full_candidates.append(w)

    for d in group_grid:
        for e in group_grid:
            f = 1.0 - d - e
            if f < -1e-9:
                continue

            w = np.array([d, e, max(0.0, f)], dtype=np.float64)
            w /= w.sum()
            qiea_candidates.append(w)

    print(
        f"\nSearching {len(full_candidates)} full-feature mixtures, "
        f"{len(qiea_candidates)} QIEA mixtures, "
        f"and {len(qiea_shares)} QIEA contribution levels...",
        flush=True,
    )

    # Each candidate group prediction is computed once.
    full_predictions = [
        val_matrix[:, :3] @ w for w in full_candidates
    ]
    qiea_predictions = [
        val_matrix[:, 3:] @ w for w in qiea_candidates
    ]

    total_combinations = (
        len(full_candidates)
        * len(qiea_candidates)
        * len(qiea_shares)
    )
    checked = 0

    for qiea_share in qiea_shares:
        full_share = 1.0 - qiea_share

        for i, full_probs in enumerate(full_predictions):
            for j, qiea_probs in enumerate(qiea_predictions):
                probs = full_share * full_probs + qiea_share * qiea_probs
                score = average_precision_score(y_val, probs)
                checked += 1

                if best is None or score > best["pr_auc"]:
                    best = {
                        "weights": np.concatenate([
                            full_candidates[i] * full_share,
                            qiea_candidates[j] * qiea_share,
                        ]),
                        "pr_auc": float(score),
                        "qiea_share": float(qiea_share),
                    }

        print(
            f"QIEA share {qiea_share:.2f} complete "
            f"({checked:,}/{total_combinations:,} combinations); "
            f"best validation PR-AUC so far: {best['pr_auc']:.6f}",
            flush=True,
        )


    weights = best["weights"]
    val_probs = val_matrix @ weights
    test_probs = test_matrix @ weights
    threshold, val_f2 = tune_threshold(y_val, val_probs)

    print("\nBEST HYBRID ENSEMBLE")
    print("Model order:", MODEL_NAMES)
    print("Weights:", [round(float(w), 3) for w in weights])
    print("QIEA total contribution:", round(best["qiea_share"], 3))
    print("Validation PR-AUC:", round(best["pr_auc"], 6))
    print("Validation F2 at selected threshold:", round(val_f2, 6))
    print("Validation-selected threshold:", threshold)

    result = get_metrics(y_test, test_probs, threshold)
    print("\nHYBRID TEST RESULTS")
    for key, value in result.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float)
              else f"{key}: {value}")

    output = (
        ROOT / "data" / "graphs" / "evaluation"
        / "phase9_qiea" / "hybrid_ensemble_result.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        weights=weights,
        threshold=threshold,
        validation_pr_auc=best["pr_auc"],
        validation_f2=val_f2,
        test_labels=y_test,
        test_probabilities=test_probs,
    )
    print(f"\nSaved result: {output}")


if __name__ == "__main__":
    main()
