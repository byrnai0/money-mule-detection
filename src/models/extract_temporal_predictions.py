from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
    / "temporal_experiment.pt"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "temporal"
)

OUTPUT_FILE = (
    MODEL_DIR
    / "temporal_predictions.csv"
)


MODELS = [
    "temporal_sage_model.pt",
    "temporal_gatv2_model.pt",
    "temporal_gin_model.pt",
]


def main():

    print("=" * 75)
    print("PHASE 7F-3 — TEMPORAL PREDICTION EXTRACTION")
    print("=" * 75)

    print("\n[1/4] Loading temporal experiment...")

    payload = torch.load(
        DATA_FILE,
        map_location="cpu",
        weights_only=False,
    )

    val_accounts = payload[
        "validation_accounts"
    ]

    test_accounts = payload[
        "test_accounts"
    ]

    val_labels = payload[
        "validation"
    ].y.numpy()

    test_labels = payload[
        "test"
    ].y.numpy()

    # ---------------------------------------------------------
    # Create separate validation/test tables.
    # ---------------------------------------------------------

    validation = pd.DataFrame(
        {
            "account_id": val_accounts,
            "is_illicit": val_labels,
        }
    )

    test = pd.DataFrame(
        {
            "account_id": test_accounts,
            "is_illicit": test_labels,
        }
    )

    print(
        f"Validation accounts: {len(validation):,}"
    )

    print(
        f"Test accounts:       {len(test):,}"
    )

    # ---------------------------------------------------------
    # Extract probabilities from saved models.
    # ---------------------------------------------------------

    print(
        "\n[2/4] Extracting temporal model predictions..."
    )

    for model_file in MODELS:

        path = MODEL_DIR / model_file

        model_payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        model_name = (
            model_file
            .replace(
                "temporal_",
                ""
            )
            .replace(
                "_model.pt",
                ""
            )
        )

        val_probabilities = np.asarray(
            model_payload[
                "validation_probabilities"
            ]
        )

        test_probabilities = np.asarray(
            model_payload[
                "test_probabilities"
            ]
        )

        print(
            f"{model_name.upper():<8} "
            f"val={len(val_probabilities):,} "
            f"test={len(test_probabilities):,}"
        )

        if len(val_probabilities) != len(
            validation
        ):
            raise RuntimeError(
                f"{model_name}: validation "
                "prediction count mismatch."
            )

        if len(test_probabilities) != len(
            test
        ):
            raise RuntimeError(
                f"{model_name}: test "
                "prediction count mismatch."
            )

        validation[
            f"{model_name}_prob"
        ] = val_probabilities

        test[
            f"{model_name}_prob"
        ] = test_probabilities

    # ---------------------------------------------------------
    # Save them separately.
    # ---------------------------------------------------------

    print(
        "\n[3/4] Creating unified temporal prediction table..."
    )

    # Accounts are period-specific, so use a period column.
    validation["period"] = "validation"
    test["period"] = "test"

    output = pd.concat(
        [
            validation,
            test,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    print(
        "\n[4/4] Running prediction checks..."
    )

    probability_columns = [
        "sage_prob",
        "gatv2_prob",
        "gin_prob",
    ]

    for column in probability_columns:

        if not output[column].between(
            0,
            1,
        ).all():

            raise RuntimeError(
                f"Invalid probability values in {column}."
            )

        print(
            f"{column:<12} "
            f"min={output[column].min():.6f} "
            f"max={output[column].max():.6f}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 75)
    print("TEMPORAL PREDICTION EXTRACTION COMPLETE")
    print("=" * 75)

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()