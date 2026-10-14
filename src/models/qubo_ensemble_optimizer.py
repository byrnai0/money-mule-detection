
from __future__ import annotations

from pathlib import Path
import json

import dimod
import neal
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "ensemble_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
    / "phase9_qubo"
)

WEIGHT_SCALE = 100
BITS_PER_WEIGHT = 7

# Penalty for violating sum(weights) == 1.
# We will explicitly filter for feasible solutions too.
CONSTRAINT_PENALTY = 0.01

NUM_READS = 2000
SEED = 42

MODEL_NAMES = ["sage", "gatv2", "gin"]

BASELINE_WEIGHTS = np.array(
    [0.78, 0.19, 0.03],
    dtype=float,
)


# ============================================================
# METRICS
# ============================================================

def evaluate(y_true, probabilities, threshold=0.50):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f2": float(
            fbeta_score(
                y_true, predictions,
                beta=2, zero_division=0
            )
        ),
        "threshold": float(threshold),
        "flagged_count": int(predictions.sum()),
    }


def ensemble_probabilities(prediction_matrix, weights):
    return prediction_matrix @ weights


# ============================================================
# QUBO CONSTRUCTION
# ============================================================

def build_qubo(
    prediction_matrix,
    labels,
    penalty=CONSTRAINT_PENALTY,
):
    """
    Encode each integer weight using 7 binary bits.

    Integer weights:
        0 <= units <= 127

    Actual weights:
        weight = units / 100

    Constraint:
        sum(integer weights) == 100

    Objective:
        Validation Brier score
        + penalty * (sum(units) - 100)^2

    The Brier score is quadratic in the binary variables,
    so this objective can be represented as a QUBO.
    """

    n_models = prediction_matrix.shape[1]

    if n_models != 3:
        raise ValueError("This implementation expects 3 models.")

    n_bits = n_models * BITS_PER_WEIGHT

    # Each variable's contribution to the ensemble probability.
    #
    # For example, a GraphSAGE bit worth 32 contributes:
    # prediction * 32 / 100.
    bit_values = np.array(
        [2**bit for model in MODEL_NAMES
         for bit in range(BITS_PER_WEIGHT)],
        dtype=float,
    )

    model_for_bit = np.repeat(
        np.arange(n_models),
        BITS_PER_WEIGHT,
    )

    coefficients = (
        prediction_matrix[:, model_for_bit]
        * bit_values[np.newaxis, :]
        / WEIGHT_SCALE
    )

    # Brier score:
    # mean((sum_j coefficients[n,j] * x_j - y_n)^2)
    #
    # Since x_j is binary, x_j^2 == x_j.
    linear_brier = np.mean(
        coefficients**2
        - 2.0 * labels[:, np.newaxis] * coefficients,
        axis=0,
    )

    # QUBO dictionary: (variable_i, variable_j) -> coefficient
    Q = {}

    variables = [
        f"b_{model}_{bit}"
        for model in MODEL_NAMES
        for bit in range(BITS_PER_WEIGHT)
    ]

    # Brier-score linear terms.
    for i, variable in enumerate(variables):
        Q[(variable, variable)] = float(
            linear_brier[i]
        )

    # Brier-score quadratic terms.
    #
    # The factor 2 comes from expanding the square.
    for i in range(n_bits):
        for j in range(i + 1, n_bits):
            coefficient = 2.0 * np.mean(
                coefficients[:, i] * coefficients[:, j]
            )

            if coefficient != 0:
                Q[(variables[i], variables[j])] = float(
                    coefficient
                )

    # Constraint penalty:
    # penalty * (sum_j bit_value[j] * x_j - 100)^2
    #
    # Expand the square and use x_j^2 == x_j.
    for i, variable in enumerate(variables):
        Q[(variable, variable)] += float(
            penalty
            * (
                bit_values[i] ** 2
                - 2 * WEIGHT_SCALE * bit_values[i]
            )
        )

    for i in range(n_bits):
        for j in range(i + 1, n_bits):
            key = (variables[i], variables[j])

            Q[key] = Q.get(key, 0.0) + float(
                2.0
                * penalty
                * bit_values[i]
                * bit_values[j]
            )

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

    return bqm, variables


# ============================================================
# DECODE AND VERIFY SOLUTIONS
# ============================================================

def decode_weights(sample):
    units = []

    for model in MODEL_NAMES:
        model_units = sum(
            (2**bit) * int(sample[f"b_{model}_{bit}"])
            for bit in range(BITS_PER_WEIGHT)
        )

        units.append(model_units)

    units = np.asarray(units, dtype=int)

    return units, units / WEIGHT_SCALE


def main():
    print("=" * 72)
    print("PHASE 9 — QUBO ENSEMBLE WEIGHT OPTIMIZATION")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}\n"
            "Run ensemble_predictions.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # Validation predictions: optimization data only
    # --------------------------------------------------------

    val_columns = [
        f"{model}_val_prob"
        for model in MODEL_NAMES
    ]

    val_mask = (
        df[val_columns].notna().all(axis=1)
    )

    val_df = df.loc[val_mask]

    X_val = val_df[val_columns].to_numpy(dtype=float)
    y_val = val_df["is_illicit"].to_numpy(dtype=int)

    if not np.isfinite(X_val).all():
        raise ValueError("Validation predictions contain invalid values.")

    print(f"\nValidation accounts: {len(y_val):,}")
    print(f"Validation positive rate: {y_val.mean():.4%}")

    # --------------------------------------------------------
    # Build and solve QUBO
    # --------------------------------------------------------

    print("\nBuilding QUBO...")

    bqm, variables = build_qubo(
        X_val,
        y_val,
    )

    print(f"Binary variables: {len(variables)}")
    print(f"QUBO interactions: {len(bqm.quadratic)}")

    print("\nRunning simulated annealing...")

    sampler = neal.SimulatedAnnealingSampler()

    sampleset = sampler.sample(
        bqm,
        num_reads=NUM_READS,
        num_sweeps=2000,
        seed=SEED,
    )

    # Do not trust the lowest-energy sample blindly.
    # Require an exact sum-to-100 feasible solution.
    feasible = []

    for sample, energy in sampleset.data(
        fields=["sample", "energy"],
        sorted_by="energy",
    ):
        units, weights = decode_weights(sample)

        if units.sum() == WEIGHT_SCALE:
            probabilities = ensemble_probabilities(
                X_val,
                weights,
            )

            score = brier_score_loss(
                y_val,
                probabilities,
            )

            feasible.append(
                (score, units, weights, energy)
            )

    if not feasible:
        raise RuntimeError(
            "No feasible solution found. "
            "Try increasing NUM_READS or adjusting "
            "CONSTRAINT_PENALTY."
        )

    # Among feasible sampled solutions, choose the one
    # with the lowest actual validation Brier score.
    feasible.sort(key=lambda item: item[0])

    (
        best_brier,
        best_units,
        best_weights,
        best_energy,
    ) = feasible[0]

    print("\nQUBO weights:")
    for name, units, weight in zip(
        MODEL_NAMES, best_units, best_weights
    ):
        print(f"  {name.upper():<8} {weight:.2f} ({units} units)")

    print(f"Weight sum: {best_weights.sum():.2f}")
    print(f"QUBO energy: {best_energy:.8f}")
    print(f"Validation Brier score: {best_brier:.8f}")
    print(f"Feasible sampled solutions: {len(feasible):,}")

    # --------------------------------------------------------
    # Compare on validation data
    # --------------------------------------------------------

    qubo_val_prob = ensemble_probabilities(
        X_val,
        best_weights,
    )

    baseline_val_prob = ensemble_probabilities(
        X_val,
        BASELINE_WEIGHTS,
    )

    val_results = {
        "qubo": evaluate(y_val, qubo_val_prob),
        "existing_0.78_0.19_0.03": evaluate(
            y_val,
            baseline_val_prob,
        ),
    }

    print("\nVALIDATION COMPARISON")
    print("-" * 72)

    for name, metrics in val_results.items():
        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric:<16}: {value}")

    # --------------------------------------------------------
    # Freeze weights, then evaluate held-out test data
    # --------------------------------------------------------

    test_columns = [
        f"{model}_test_prob"
        for model in MODEL_NAMES
    ]

    test_mask = (
        df[test_columns].notna().all(axis=1)
    )

    test_df = df.loc[test_mask]

    X_test = test_df[test_columns].to_numpy(dtype=float)
    y_test = test_df["is_illicit"].to_numpy(dtype=int)

    if not np.isfinite(X_test).all():
        raise ValueError("Test predictions contain invalid values.")

    qubo_test_prob = ensemble_probabilities(
        X_test,
        best_weights,
    )

    baseline_test_prob = ensemble_probabilities(
        X_test,
        BASELINE_WEIGHTS,
    )

    test_results = {
        "qubo": evaluate(y_test, qubo_test_prob),
        "existing_0.78_0.19_0.03": evaluate(
            y_test,
            baseline_test_prob,
        ),
    }

    print("\n" + "=" * 72)
    print("HELD-OUT TEST COMPARISON")
    print("=" * 72)

    for name, metrics in test_results.items():
        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric:<16}: {value}")

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "method": "QUBO with simulated annealing",
        "qubo_objective": "validation Brier score",
        "constraint": "nonnegative weights in 0.01 increments summing to 1",
        "model_names": MODEL_NAMES,
        "qubo_weights": {
            name: float(weight)
            for name, weight in zip(MODEL_NAMES, best_weights)
        },
        "baseline_weights": {
            name: float(weight)
            for name, weight in zip(
                MODEL_NAMES, BASELINE_WEIGHTS
            )
        },
        "qubo_energy": float(best_energy),
        "validation": val_results,
        "test": test_results,
        "configuration": {
            "num_reads": NUM_READS,
            "num_sweeps": 2000,
            "seed": SEED,
            "constraint_penalty": CONSTRAINT_PENALTY,
        },
        "warning": (
            "The QUBO optimizes Brier score, not PR-AUC directly. "
            "Simulated annealing is heuristic and does not guarantee "
            "a global optimum."
        ),
    }

    report_file = OUTPUT_DIR / "qubo_ensemble_report.json"

    with report_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    weight_file = OUTPUT_DIR / "qubo_weights.csv"

    pd.DataFrame(
        {
            "model": MODEL_NAMES,
            "weight_units": best_units,
            "qubo_weight": best_weights,
            "baseline_weight": BASELINE_WEIGHTS,
        }
    ).to_csv(weight_file, index=False)

    print("\nSaved artifacts:")
    print(f"  Report:  {report_file}")
    print(f"  Weights: {weight_file}")

    print("\nPhase 9 QUBO experiment complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
