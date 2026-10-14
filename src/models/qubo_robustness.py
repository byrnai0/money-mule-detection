
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import neal

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score,
)

from qubo_ensemble_optimizer import (
    INPUT_FILE,
    MODEL_NAMES,
    WEIGHT_SCALE,
    build_qubo,
    decode_weights,
    ensemble_probabilities,
)


OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "graphs"
    / "evaluation"
    / "phase9_qubo"
)

PENALTIES = [0.001, 0.01, 0.1, 1.0]
SEEDS = [11, 22, 33, 44, 55]

NUM_READS = 500
NUM_SWEEPS = 1000


def main():
    print("=" * 72)
    print("PHASE 9 — QUBO ROBUSTNESS EXPERIMENT")
    print("=" * 72)

    df = pd.read_csv(INPUT_FILE)

    columns = [
        f"{model}_val_prob"
        for model in MODEL_NAMES
    ]

    mask = df[columns].notna().all(axis=1)
    val_df = df.loc[mask]

    X_val = val_df[columns].to_numpy(dtype=float)
    y_val = val_df["is_illicit"].to_numpy(dtype=int)

    rows = []

    for penalty in PENALTIES:
        print(f"\nConstraint penalty: {penalty}")

        bqm, _ = build_qubo(
            X_val,
            y_val,
            penalty=penalty,
        )

        for seed in SEEDS:
            sampler = neal.SimulatedAnnealingSampler()

            sampleset = sampler.sample(
                bqm,
                num_reads=NUM_READS,
                num_sweeps=NUM_SWEEPS,
                seed=seed,
            )

            best = None

            for sample, energy in sampleset.data(
                fields=["sample", "energy"],
                sorted_by="energy",
            ):
                units, weights = decode_weights(sample)

                if units.sum() != WEIGHT_SCALE:
                    continue

                probabilities = ensemble_probabilities(
                    X_val,
                    weights,
                )

                brier = brier_score_loss(
                    y_val,
                    probabilities,
                )

                if best is None or brier < best["brier_score"]:
                    predictions = (
                        probabilities >= 0.5
                    ).astype(int)

                    best = {
                        "weights": weights.copy(),
                        "units": units.copy(),
                        "brier_score": float(brier),
                        "pr_auc": float(
                            average_precision_score(
                                y_val, probabilities
                            )
                        ),
                        "precision_at_0.5": float(
                            precision_score(
                                y_val,
                                predictions,
                                zero_division=0,
                            )
                        ),
                        "recall_at_0.5": float(
                            recall_score(
                                y_val,
                                predictions,
                                zero_division=0,
                            )
                        ),
                        "f2_at_0.5": float(
                            fbeta_score(
                                y_val,
                                predictions,
                                beta=2,
                                zero_division=0,
                            )
                        ),
                        "qubo_energy": float(energy),
                    }

            if best is None:
                print(f"  Seed {seed}: no feasible solution")
                rows.append({
                    "penalty": penalty,
                    "seed": seed,
                    "feasible": False,
                })
                continue

            row = {
                "penalty": penalty,
                "seed": seed,
                "feasible": True,
                "weight_sum": int(best["units"].sum()),
                "brier_score": best["brier_score"],
                "pr_auc": best["pr_auc"],
                "precision_at_0.5": best["precision_at_0.5"],
                "recall_at_0.5": best["recall_at_0.5"],
                "f2_at_0.5": best["f2_at_0.5"],
                "qubo_energy": best["qubo_energy"],
            }

            for name, weight in zip(
                MODEL_NAMES,
                best["weights"],
            ):
                row[f"{name}_weight"] = float(weight)

            rows.append(row)

            print(
                f"  Seed {seed}: "
                f"weights={best['weights'].round(2).tolist()}, "
                f"Brier={best['brier_score']:.6f}, "
                f"PR-AUC={best['pr_auc']:.6f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)

    output_file = OUTPUT_DIR / "qubo_robustness.csv"
    results.to_csv(output_file, index=False)

    valid = results[
        results["feasible"] == True
    ].copy()

    print("\n" + "=" * 72)
    print("ROBUSTNESS SUMMARY")
    print("=" * 72)

    if valid.empty:
        print("No feasible solutions were found.")
    else:
        summary = valid.groupby("penalty").agg(
            successful_runs=("feasible", "sum"),
            mean_brier=("brier_score", "mean"),
            best_brier=("brier_score", "min"),
            mean_pr_auc=("pr_auc", "mean"),
            best_pr_auc=("pr_auc", "max"),
            mean_f2=("f2_at_0.5", "mean"),
        )

        print(summary.to_string())

        print("\nWeight variability:")
        weight_columns = [
            f"{name}_weight"
            for name in MODEL_NAMES
        ]

        print(
            valid.groupby("penalty")[weight_columns]
            .agg(["mean", "std"])
            .round(3)
            .to_string()
        )

    print(f"\nSaved: {output_file}")
    print("=" * 72)


if __name__ == "__main__":
    main()
