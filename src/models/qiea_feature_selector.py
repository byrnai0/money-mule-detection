
"""
Quantum-inspired evolutionary feature selector.

The algorithm runs on a classical computer. It uses Q-bit-inspired
amplitudes, binary sampling, elitism, and rotation-based updates.

The evaluator must return a dictionary containing:
    {"pr_auc": float, "f2": float}

No GNN training or graph loading happens in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


FitnessEvaluator = Callable[[np.ndarray], dict[str, float]]


@dataclass
class QIEAConfig:
    population_size: int = 12
    generations: int = 15
    min_features: int = 5
    max_features: int = 22
    rotation_step: float = 0.03 * np.pi
    min_angle: float = 0.03
    random_seed: int = 42


class QIEAFeatureSelector:
    def __init__(
        self,
        n_features: int,
        evaluator: FitnessEvaluator,
        config: QIEAConfig | None = None,
    ):
        self.n_features = n_features
        self.evaluator = evaluator
        self.config = config or QIEAConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        if n_features < 1:
            raise ValueError("n_features must be positive.")

        if not (
            1 <= self.config.min_features
            <= self.config.max_features
            <= n_features
        ):
            raise ValueError("Invalid feature-count limits.")

        if self.config.population_size < 2:
            raise ValueError("Population size must be at least 2.")

        # Each feature has amplitudes:
        # alpha = cos(theta), beta = sin(theta)
        # P(feature selected) = beta**2 = sin(theta)**2
        self.theta = np.full(n_features, np.pi / 4, dtype=float)

        self.best_mask: np.ndarray | None = None
        self.best_fitness = float("-inf")
        self.best_metrics: dict[str, float] | None = None

        # Avoid reevaluating identical subsets.
        self.cache: dict[tuple[int, ...], dict[str, float]] = {}
        self.history: list[dict] = []

    def _sample_candidate(self) -> np.ndarray:
        probabilities = np.sin(self.theta) ** 2
        mask = (
            self.rng.random(self.n_features) < probabilities
        ).astype(np.int8)

        selected = int(mask.sum())

        # Repair candidates that violate cardinality constraints.
        if selected < self.config.min_features:
            zeros = np.flatnonzero(mask == 0)
            add_count = self.config.min_features - selected
            chosen = self.rng.choice(
                zeros, size=add_count, replace=False
            )
            mask[chosen] = 1

        elif selected > self.config.max_features:
            ones = np.flatnonzero(mask == 1)
            remove_count = selected - self.config.max_features
            chosen = self.rng.choice(
                ones, size=remove_count, replace=False
            )
            mask[chosen] = 0

        return mask

    def _evaluate(self, mask: np.ndarray) -> dict[str, float]:
        key = tuple(int(v) for v in mask)

        if key not in self.cache:
            result = self.evaluator(mask.copy())
            pr_auc = float(result["pr_auc"])
            f2 = float(result["f2"])

            if not np.isfinite(pr_auc) or not np.isfinite(f2):
                raise ValueError("Evaluator returned non-finite metrics.")

            self.cache[key] = {
                "pr_auc": pr_auc,
                "f2": f2,
            }

        return self.cache[key]

    def _update_toward_best(self, best_mask: np.ndarray) -> None:
        """
        Rotate each Q-bit-inspired angle toward the current best
        binary solution. P(1) = sin(theta)^2.

        This is a simplified, explicitly documented rotation rule,
        not a quantum circuit or physical quantum operation.
        """
        step = self.config.rotation_step

        for i, target_bit in enumerate(best_mask):
            p_one = np.sin(self.theta[i]) ** 2

            # Skip updates when the probability already strongly
            # favors the target bit.
            if target_bit == 1 and p_one >= 0.99:
                continue
            if target_bit == 0 and p_one <= 0.01:
                continue

            self.theta[i] += step if target_bit == 1 else -step

        # Keep probabilities away from exact 0 and 1 so exploration
        # remains possible in later generations.
        upper = (np.pi / 2) - self.config.min_angle
        self.theta = np.clip(
            self.theta,
            self.config.min_angle,
            upper,
        )

    def run(self) -> dict:
        for generation in range(self.config.generations):
            candidates = [
                self._sample_candidate()
                for _ in range(self.config.population_size)
            ]

            # Preserve the global best through elitism.
            if self.best_mask is not None:
                candidates[0] = self.best_mask.copy()

            evaluated = []
            for mask in candidates:
                metrics = self._evaluate(mask)
                evaluated.append((mask, metrics))

                if metrics["pr_auc"] > self.best_fitness:
                    self.best_fitness = metrics["pr_auc"]
                    self.best_mask = mask.copy()
                    self.best_metrics = metrics.copy()

            # Update the Q-bit-inspired search state toward the
            # best candidate found so far.
            self._update_toward_best(self.best_mask)

            self.history.append({
                "generation": generation + 1,
                "generation_best_pr_auc": max(
                    m["pr_auc"] for _, m in evaluated
                ),
                "global_best_pr_auc": self.best_fitness,
                "global_best_f2": self.best_metrics["f2"],
                "selected_features": int(self.best_mask.sum()),
                "unique_subsets_evaluated": len(self.cache),
            })

        return {
            "best_mask": self.best_mask.copy(),
            "selected_indices": np.flatnonzero(self.best_mask),
            "selected_count": int(self.best_mask.sum()),
            "best_metrics": self.best_metrics.copy(),
            "history": self.history.copy(),
            "unique_subsets_evaluated": len(self.cache),
            "final_selection_probabilities": (
                np.sin(self.theta) ** 2
            ),
        }
