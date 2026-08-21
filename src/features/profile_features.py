from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_features_v2.csv"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "account_labels.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_profile.csv"
)


def main() -> None:

    print("=" * 75)
    print("PHASE 4 — FEATURE PROFILING")
    print("=" * 75)

    print("\nLoading features...")
    features = pd.read_csv(FEATURE_FILE)

    print("Loading account labels...")
    labels = pd.read_csv(LABEL_FILE)

    print(f"Feature rows: {len(features):,}")
    print(f"Label rows:   {len(labels):,}")

    # ---------------------------------------------------------
    # Merge only for analysis.
    # Labels are NOT part of the feature matrix.
    # ---------------------------------------------------------

    df = features.merge(
        labels[
            [
                "account_id",
                "is_illicit",
            ]
        ],
        on="account_id",
        how="left",
        validate="one_to_one",
    )

    if df["is_illicit"].isna().any():
        raise RuntimeError(
            "Some accounts have no corresponding label."
        )

    feature_columns = [
        col
        for col in features.columns
        if col != "account_id"
    ]

    print(
        f"\nAnalyzing {len(feature_columns)} candidate features..."
    )

    # ---------------------------------------------------------
    # Basic distribution profile
    # ---------------------------------------------------------

    rows = []

    for feature in feature_columns:

        series = df[feature]

        rows.append(
            {
                "feature": feature,
                "dtype": str(series.dtype),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "zero_pct": float(
                    (series == 0).mean() * 100
                ),
                "missing": int(series.isna().sum()),
                "unique_values": int(series.nunique()),
                "illicit_mean": float(
                    df.loc[
                        df["is_illicit"] == 1,
                        feature
                    ].mean()
                ),
                "legit_mean": float(
                    df.loc[
                        df["is_illicit"] == 0,
                        feature
                    ].mean()
                ),
            }
        )

    profile = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Effect / separation metric
    #
    # Difference between illicit and legitimate means,
    # normalized by overall standard deviation.
    #
    # This is NOT a statistical significance test.
    # It is only a simple screening signal.
    # ---------------------------------------------------------

    profile["mean_difference"] = (
        profile["illicit_mean"]
        - profile["legit_mean"]
    )

    profile["standardized_mean_difference"] = np.where(
        profile["std"] > 0,
        profile["mean_difference"]
        / profile["std"],
        0.0,
    )

    # ---------------------------------------------------------
    # Correlation with target
    # ---------------------------------------------------------

    correlations = {}

    for feature in feature_columns:
        correlations[feature] = df[
            [feature, "is_illicit"]
        ].corr(numeric_only=True).iloc[0, 1]

    profile["target_correlation"] = (
        profile["feature"]
        .map(correlations)
    )

    # ---------------------------------------------------------
    # Sort by target association for inspection
    # ---------------------------------------------------------

    profile = profile.sort_values(
        by="target_correlation",
        key=lambda s: s.abs(),
        ascending=False,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 75)
    print("FEATURE PROFILE")
    print("=" * 75)

    pd.set_option(
        "display.max_rows",
        len(profile)
    )

    pd.set_option(
        "display.max_columns",
        None
    )

    print(
        profile[
            [
                "feature",
                "min",
                "max",
                "mean",
                "median",
                "std",
                "zero_pct",
                "unique_values",
                "illicit_mean",
                "legit_mean",
                "standardized_mean_difference",
                "target_correlation",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Highly correlated feature pairs
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("HIGHLY CORRELATED FEATURE PAIRS")
    print("=" * 75)

    corr = df[
        feature_columns
    ].corr()

    high_corr_pairs = []

    for i, feature_a in enumerate(feature_columns):
        for feature_b in feature_columns[i + 1:]:
            value = corr.loc[
                feature_a,
                feature_b
            ]

            if abs(value) >= 0.95:
                high_corr_pairs.append(
                    (
                        feature_a,
                        feature_b,
                        value,
                    )
                )

    if high_corr_pairs:
        for a, b, value in sorted(
            high_corr_pairs,
            key=lambda x: abs(x[2]),
            reverse=True,
        ):
            print(
                f"{a:<40} "
                f"{b:<40} "
                f"{value:.4f}"
            )
    else:
        print("No feature pairs above |correlation| >= 0.95")

    print("\n" + "=" * 75)
    print(f"Profile saved to:\n{OUTPUT_FILE}")
    print("=" * 75)


if __name__ == "__main__":
    main()