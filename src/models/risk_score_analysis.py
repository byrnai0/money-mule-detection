from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "tuned_voting_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "evaluation"
)

PLOT_FILE = (
    OUTPUT_DIR
    / "risk_score_distribution.png"
)

STATS_FILE = (
    OUTPUT_DIR
    / "risk_score_statistics.csv"
)

THRESHOLD = 0.79


def summarize_scores(
    scores: pd.Series,
) -> dict:

    return {
        "count": int(scores.shape[0]),
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "std": float(scores.std()),
        "p90": float(scores.quantile(0.90)),
        "p95": float(scores.quantile(0.95)),
        "p99": float(scores.quantile(0.99)),
    }


def main() -> None:

    print("=" * 75)
    print("PHASE 7D — RISK SCORE ANALYSIS")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load test predictions
    # ---------------------------------------------------------

    print("\n[1/4] Loading test risk scores...")

    df = pd.read_csv(
        PREDICTION_FILE,
        usecols=[
            "account_id",
            "is_illicit",
            "tuned_vote_test_prob",
        ],
    )

    df = df[
        df["tuned_vote_test_prob"].notna()
    ].copy()

    print(
        f"Test accounts: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Split classes
    # ---------------------------------------------------------

    print(
        "\n[2/4] Separating legitimate and illicit accounts..."
    )

    legitimate = df.loc[
        df["is_illicit"] == 0,
        "tuned_vote_test_prob",
    ]

    illicit = df.loc[
        df["is_illicit"] == 1,
        "tuned_vote_test_prob",
    ]

    legit_stats = summarize_scores(
        legitimate
    )

    illicit_stats = summarize_scores(
        illicit
    )

    print("\nLEGITIMATE ACCOUNTS")

    for key, value in legit_stats.items():

        if key == "count":
            print(
                f"{key:<10}: {value:,}"
            )
        else:
            print(
                f"{key:<10}: {value:.6f}"
            )

    print("\nILLICIT ACCOUNTS")

    for key, value in illicit_stats.items():

        if key == "count":
            print(
                f"{key:<10}: {value:,}"
            )
        else:
            print(
                f"{key:<10}: {value:.6f}"
            )

    # ---------------------------------------------------------
    # Threshold statistics
    # ---------------------------------------------------------

    print(
        "\n[3/4] Checking score separation at threshold..."
    )

    legit_above = int(
        (legitimate >= THRESHOLD).sum()
    )

    illicit_above = int(
        (illicit >= THRESHOLD).sum()
    )

    legit_below = int(
        (legitimate < THRESHOLD).sum()
    )

    illicit_below = int(
        (illicit < THRESHOLD).sum()
    )

    print(
        f"\nThreshold: {THRESHOLD:.2f}"
    )

    print(
        f"Legitimate >= threshold: "
        f"{legit_above:,}"
    )

    print(
        f"Illicit >= threshold:     "
        f"{illicit_above:,}"
    )

    print(
        f"Legitimate < threshold:  "
        f"{legit_below:,}"
    )

    print(
        f"Illicit < threshold:      "
        f"{illicit_below:,}"
    )

    # ---------------------------------------------------------
    # Save statistics
    # ---------------------------------------------------------

    stats_rows = []

    for class_name, stats in [
        ("legitimate", legit_stats),
        ("illicit", illicit_stats),
    ]:

        row = {
            "class": class_name,
            **stats,
            "threshold": THRESHOLD,
            "above_threshold": (
                legit_above
                if class_name == "legitimate"
                else illicit_above
            ),
            "below_threshold": (
                legit_below
                if class_name == "legitimate"
                else illicit_below
            ),
        }

        stats_rows.append(row)

    stats_df = pd.DataFrame(
        stats_rows
    )

    stats_df.to_csv(
        STATS_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    print(
        "\n[4/4] Creating risk-score distribution plot..."
    )

    plt.figure(
        figsize=(10, 6)
    )

    # Use the same bins for both groups.
    bins = np.linspace(
        0,
        1,
        51,
    )

    plt.hist(
        legitimate,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Legitimate",
    )

    plt.hist(
        illicit,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Illicit",
    )

    plt.axvline(
        THRESHOLD,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {THRESHOLD:.2f}",
    )

    plt.xlabel(
        "Predicted Risk Score"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Risk Score Distribution — Tuned Soft Voting"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=200,
    )

    plt.close()

    print("\n" + "=" * 75)
    print("PHASE 7D COMPLETE")
    print("=" * 75)

    print(
        f"\nPlot:\n{PLOT_FILE}"
    )

    print(
        f"\nStatistics:\n{STATS_FILE}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()