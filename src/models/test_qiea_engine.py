
import numpy as np

from qiea_feature_selector import (
    QIEAConfig,
    QIEAFeatureSelector,
)


def toy_evaluator(mask: np.ndarray) -> dict[str, float]:
    # Synthetic objective: features 0, 2, 4, 6 are useful.
    useful = {0, 2, 4, 6}
    selected = set(np.flatnonzero(mask))

    overlap = len(selected & useful) / len(useful)
    unnecessary = len(selected - useful) / len(mask)

    score = max(0.0, min(1.0, 0.5 + 0.5 * overlap - unnecessary))
    return {"pr_auc": score, "f2": score}


if __name__ == "__main__":
    config = QIEAConfig(
        population_size=12,
        generations=15,
        min_features=2,
        max_features=7,
        random_seed=42,
    )

    optimizer = QIEAFeatureSelector(
        n_features=8,
        evaluator=toy_evaluator,
        config=config,
    )

    result = optimizer.run()

    print("Selected indices:", result["selected_indices"])
    print("Selected count:", result["selected_count"])
    print("Best metrics:", result["best_metrics"])
    print("Unique subsets evaluated:",
          result["unique_subsets_evaluated"])
