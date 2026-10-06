"""
Thresholding strategies for Autoencoder-based IDS.
- Fixed threshold
- Statistical threshold (mean + k*std)
- PSO-optimized threshold
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from typing import Dict, Tuple, Callable, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed Threshold
# ─────────────────────────────────────────────────────────────────────────────

class FixedThreshold:
    """Fixed threshold strategy."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def compute(self, reconstruction_errors: np.ndarray) -> float:
        """Return fixed threshold."""
        return self.threshold

    def classify(self, reconstruction_errors: np.ndarray) -> np.ndarray:
        """Classify samples as normal (0) or anomaly (1)."""
        return (reconstruction_errors > self.threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Threshold
# ─────────────────────────────────────────────────────────────────────────────

class StatisticalThreshold:
    """Statistical threshold: mean + k * std."""

    def __init__(self, k: float = 1.5):
        """
        k: number of standard deviations from mean
        Higher k = more conservative (fewer anomalies detected)
        Lower k = more aggressive (more anomalies detected)
        Default 1.5 balances sensitivity and specificity
        """
        self.k = k
        self.threshold = None
        self.mean_ = None
        self.std_ = None

    def compute(self, reconstruction_errors: np.ndarray) -> float:
        """Compute threshold from training errors."""
        self.mean_ = np.mean(reconstruction_errors)
        self.std_ = np.std(reconstruction_errors)
        self.threshold = self.mean_ + self.k * self.std_
        logger.info(f"[Statistical Threshold] Mean: {self.mean_:.4f}, "
                   f"Std: {self.std_:.4f}, Threshold: {self.threshold:.4f}")
        return self.threshold

    def classify(self, reconstruction_errors: np.ndarray) -> np.ndarray:
        """Classify samples."""
        return (reconstruction_errors > self.threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# PSO-Optimized Threshold
# ─────────────────────────────────────────────────────────────────────────────

class PSOOptimizedThreshold:
    """PSO-based threshold optimization for F1-score maximization."""

    def __init__(self, n_particles: int = 30, n_iter: int = 20,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.threshold = None
        self.best_score_ = -np.inf

    def _compute_f1(self, threshold: float, errors: np.ndarray,
                    y_true: np.ndarray) -> float:
        """Compute F1-score for a threshold."""
        y_pred = (errors > threshold).astype(int)
        if len(np.unique(y_pred)) == 1:
            return 0.0
        return f1_score(y_true, y_pred)

    def compute(self, reconstruction_errors: np.ndarray,
                y_true: np.ndarray) -> float:
        """Compute optimal threshold using PSO."""
        min_error = reconstruction_errors.min()
        max_error = reconstruction_errors.max()

        # Initialize particles
        particles = np.random.uniform(
            min_error, max_error, size=(self.n_particles,)
        )
        velocities = np.random.uniform(-max_error, max_error, size=(self.n_particles,))
        fitness = np.zeros(self.n_particles)

        # Evaluate initial population
        for i in range(self.n_particles):
            fitness[i] = self._compute_f1(particles[i], reconstruction_errors, y_true)

        pbest = particles.copy()
        pbest_fitness = fitness.copy()
        gbest_idx = np.argmax(fitness)
        gbest = particles[gbest_idx]
        gbest_fitness = fitness[gbest_idx]

        logger.info(f"[PSO Threshold] Optimizing threshold, "
                   f"Error range: [{min_error:.4f}, {max_error:.4f}]")

        # PSO iterations
        for iteration in range(self.n_iter):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (pbest[i] - particles[i]) +
                    self.c2 * r2 * (gbest - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], min_error, max_error)
                fitness[i] = self._compute_f1(particles[i], reconstruction_errors, y_true)

                if fitness[i] > pbest_fitness[i]:
                    pbest[i] = particles[i]
                    pbest_fitness[i] = fitness[i]

                    if fitness[i] > gbest_fitness:
                        gbest = particles[i]
                        gbest_fitness = fitness[i]

            logger.info(f"[PSO Threshold] Iteration {iteration + 1}/{self.n_iter}, "
                       f"Best F1: {gbest_fitness:.4f}, Threshold: {gbest:.4f}")

        self.threshold = gbest
        self.best_score_ = gbest_fitness
        logger.info(f"[PSO Threshold] Final threshold: {self.threshold:.4f}, "
                   f"F1-score: {self.best_score_:.4f}")
        return self.threshold

    def classify(self, reconstruction_errors: np.ndarray) -> np.ndarray:
        """Classify samples."""
        return (reconstruction_errors > self.threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Factory
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdFactory:
    """Factory for creating threshold strategies."""

    @staticmethod
    def create_threshold(method: str, **kwargs):
        """Create threshold strategy."""
        if method.lower() == 'fixed':
            return FixedThreshold(threshold=kwargs.get('threshold', 0.5))
        elif method.lower() == 'statistical':
            return StatisticalThreshold(k=kwargs.get('k', 3.0))
        elif method.lower() == 'pso':
            return PSOOptimizedThreshold(
                n_particles=kwargs.get('n_particles', 30),
                n_iter=kwargs.get('n_iter', 20)
            )
        else:
            raise ValueError(f"Unknown threshold method: {method}")
