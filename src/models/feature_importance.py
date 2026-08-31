from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from architectures import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "training_graph.pt"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "models"
    / "sampled"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "global_feature_importance.csv"
)

MODEL_WEIGHTS = {
    "sage": 0.78,
    "gatv2": 0.19,
    "gin": 0.03,
}

BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]

SEED = 42


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = SEED):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Load models
# =========================================================

def load_models(input_dim: int):

    models = {}

    for name in MODEL_WEIGHTS:

        model_file = (
            MODEL_DIR
            / f"{name}_sampled_model.pt"
        )

        if not model_file.exists():

            raise FileNotFoundError(
                f"Missing model:\n{model_file}"
            )

        payload = torch.load(
            model_file,
            map_location="cpu",
            weights_only=False,
        )

        model = build_model(
            name=name,
            in_channels=input_dim,
            hidden_channels=64,
            dropout=0.30,
        )

        model.load_state_dict(
            payload["model_state_dict"]
        )

        model = model.to(
            DEVICE
        )

        model.eval()

        models[name] = model

    return models


# =========================================================
# Prediction
# =========================================================

def predict_model(
    model,
    data,
    input_mask,
):

    # Make the sampler deterministic as far as practical.
    set_seed()

    loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=input_mask,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    probabilities = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(
                DEVICE
            )

            logits = model(
                batch.x,
                batch.edge_index,
            )

            seed_logits = logits[
                :batch.batch_size
            ]

            probs = torch.softmax(
                seed_logits,
                dim=1,
            )[:, 1]

            probabilities.extend(
                probs.cpu().numpy()
            )

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )


# =========================================================
# Ensemble
# =========================================================

def ensemble_probability(
    predictions: dict[str, np.ndarray],
):

    return (
        MODEL_WEIGHTS["sage"]
        * predictions["sage"]
        +
        MODEL_WEIGHTS["gatv2"]
        * predictions["gatv2"]
        +
        MODEL_WEIGHTS["gin"]
        * predictions["gin"]
    )


# =========================================================
# Main
# =========================================================

def main():

    start = time.perf_counter()

    print("=" * 75)
    print("PHASE 8A — GLOBAL FEATURE IMPORTANCE")
    print("=" * 75)

    print(
        "\nExplainability method:"
        "\n  Permutation importance on validation data"
        "\n  Primary metric: PR-AUC"
    )

    print(
        "\nEnsemble weights:"
        f"\n  GraphSAGE = {MODEL_WEIGHTS['sage']:.2f}"
        f"\n  GATv2     = {MODEL_WEIGHTS['gatv2']:.2f}"
        f"\n  GIN       = {MODEL_WEIGHTS['gin']:.2f}"
    )

    # ---------------------------------------------------------
    # Load training graph
    # ---------------------------------------------------------

    print(
        "\n[1/5] Loading training graph..."
    )

    payload = torch.load(
        DATA_FILE,
        map_location="cpu",
        weights_only=False,
    )

    original_data = payload["data"]

    print(
        f"Nodes: {original_data.num_nodes:,}"
    )

    print(
        f"Features: {original_data.x.shape[1]}"
    )

    # ---------------------------------------------------------
    # Feature names
    # ---------------------------------------------------------

    print(
        "\n[2/5] Loading feature names..."
    )

    feature_names = None

    if MANIFEST_FILE.exists():

        manifest = pd.read_csv(
            MANIFEST_FILE
        )

        feature_names = (
            manifest[
                "feature_name"
            ]
            .tolist()
        )

    if (
        feature_names is None
        or len(feature_names)
        != original_data.x.shape[1]
    ):

        # Fallback: use the feature order
        # stored/generated in the graph.
        feature_names = [
            f"feature_{i}"
            for i in range(
                original_data.x.shape[1]
            )
        ]

        print(
            "Manifest unavailable/mismatched; "
            "using generic feature names."
        )

    print(
        f"Features loaded: "
        f"{len(feature_names)}"
    )

    # ---------------------------------------------------------
    # Validation labels
    # ---------------------------------------------------------

    validation_mask = (
        original_data.val_mask
    )

    y_validation = (
        original_data.y[
            validation_mask
        ]
        .numpy()
        .astype(np.int64)
    )

    print(
        f"Validation accounts: "
        f"{len(y_validation):,}"
    )

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------

    print(
        "\n[3/5] Loading GNN models..."
    )

    models = load_models(
        original_data.x.shape[1]
    )

    print(
        "Loaded: "
        + ", ".join(
            name.upper()
            for name in models
        )
    )

    # ---------------------------------------------------------
    # Baseline predictions
    # ---------------------------------------------------------

    print(
        "\n[4/5] Calculating baseline predictions..."
    )

    baseline_predictions = {}

    for name, model in models.items():

        print(
            f"  Predicting with {name.upper()}..."
        )

        baseline_predictions[name] = (
            predict_model(
                model,
                original_data,
                validation_mask,
            )
        )

        print(
            f"    predictions: "
            f"{len(baseline_predictions[name]):,}"
        )

    baseline_ensemble = (
        ensemble_probability(
            baseline_predictions
        )
    )

    baseline_pr_auc = (
        average_precision_score(
            y_validation,
            baseline_ensemble,
        )
    )

    baseline_roc_auc = (
        roc_auc_score(
            y_validation,
            baseline_ensemble,
        )
    )

    print(
        "\nBaseline ensemble:"
    )

    print(
        f"  PR-AUC:  "
        f"{baseline_pr_auc:.6f}"
    )

    print(
        f"  ROC-AUC: "
        f"{baseline_roc_auc:.6f}"
    )

    # ---------------------------------------------------------
    # Permutation importance
    # ---------------------------------------------------------

    print(
        "\n[5/5] Permuting features..."
    )

    x_original = (
        original_data.x
        .cpu()
        .numpy()
        .copy()
    )

    rng = np.random.default_rng(
        SEED
    )

    results = []

    total_features = len(
        feature_names
    )

    for feature_index, feature_name in enumerate(
        feature_names,
        start=1,
    ):

        feature_start = time.perf_counter()

        # -----------------------------------------------------
        # Shuffle ONE feature across all nodes.
        # -----------------------------------------------------

        shuffled_x = (
            x_original.copy()
        )

        shuffled_x[:, feature_index - 1] = (
            rng.permutation(
                shuffled_x[
                    :,
                    feature_index - 1,
                ]
            )
        )

        modified_data = Data(
            x=torch.from_numpy(
                shuffled_x
            ),

            edge_index=(
                original_data.edge_index
            ),

            y=(
                original_data.y
            ),

            val_mask=(
                original_data.val_mask
            ),
        )

        # -----------------------------------------------------
        # Run all three models.
        # -----------------------------------------------------

        perturbed_predictions = {}

        for name, model in models.items():

            perturbed_predictions[name] = (
                predict_model(
                    model,
                    modified_data,
                    validation_mask,
                )
            )

        perturbed_ensemble = (
            ensemble_probability(
                perturbed_predictions
            )
        )

        perturbed_pr_auc = (
            average_precision_score(
                y_validation,
                perturbed_ensemble,
            )
        )

        perturbed_roc_auc = (
            roc_auc_score(
                y_validation,
                perturbed_ensemble,
            )
        )

        pr_auc_drop = (
            baseline_pr_auc
            - perturbed_pr_auc
        )

        roc_auc_drop = (
            baseline_roc_auc
            - perturbed_roc_auc
        )

        elapsed = (
            time.perf_counter()
            - feature_start
        )

        results.append(
            {
                "feature": feature_name,
                "baseline_pr_auc": (
                    baseline_pr_auc
                ),
                "permuted_pr_auc": (
                    perturbed_pr_auc
                ),
                "pr_auc_drop": (
                    pr_auc_drop
                ),
                "baseline_roc_auc": (
                    baseline_roc_auc
                ),
                "permuted_roc_auc": (
                    perturbed_roc_auc
                ),
                "roc_auc_drop": (
                    roc_auc_drop
                ),
                "feature_rank_pending": True,
            }
        )

        print(
            f"[{feature_index:02d}/{total_features}] "
            f"{feature_name:<45} "
            f"PR-AUC drop={pr_auc_drop:+.6f} "
            f"({elapsed:.1f}s)"
        )

    # ---------------------------------------------------------
    # Rank features
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "pr_auc_drop",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    results_df[
        "feature_rank"
    ] = np.arange(
        1,
        len(results_df) + 1,
    )

    results_df[
        "feature_importance_relative"
    ] = (
        results_df[
            "pr_auc_drop"
        ]
        .clip(
            lower=0
        )
    )

    total_importance = (
        results_df[
            "feature_importance_relative"
        ].sum()
    )

    if total_importance > 0:

        results_df[
            "importance_share"
        ] = (
            results_df[
                "feature_importance_relative"
            ]
            / total_importance
        )

    else:

        results_df[
            "importance_share"
        ] = 0.0

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("TOP GLOBAL FEATURES")
    print("=" * 75)

    print(
        results_df[
            [
                "feature_rank",
                "feature",
                "pr_auc_drop",
                "roc_auc_drop",
                "importance_share",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    total_elapsed = (
        time.perf_counter()
        - start
    )

    print("\n" + "=" * 75)
    print("PHASE 8A COMPLETE")
    print("=" * 75)

    print(
        f"\nBaseline PR-AUC: "
        f"{baseline_pr_auc:.6f}"
    )

    print(
        f"Baseline ROC-AUC: "
        f"{baseline_roc_auc:.6f}"
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Runtime: "
        f"{total_elapsed:.2f}s"
    )

    print("=" * 75)


if __name__ == "__main__":
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    main()