"""
Feature Selection module for IDS pipeline.

Methods:
  Filter    – statistical tests (MI, ANOVA F)
  Wrapper   – Recursive Feature Elimination (RFE)
  Embedded  – Random Forest feature importance
  PSO       – Particle Swarm Optimization
  ACO       – Ant Colony Optimization
"""

import numpy as np
from sklearn.feature_selection import (
    f_classif, mutual_info_classif, RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_k(k: int, n_features: int) -> int:
    return max(1, min(k, n_features))


def _score_subset(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  mask: np.ndarray) -> float:
    """Quick RF accuracy on a feature subset – used by wrapper methods."""
    if mask.sum() == 0:
        return 0.0
    clf = RandomForestClassifier(n_estimators=30, max_depth=6,
                                  n_jobs=-1, random_state=42)
    clf.fit(X_train[:, mask], y_train)
    return clf.score(X_val[:, mask], y_val)


# ─────────────────────────────────────────────────────────────────────────────
# Filter Methods
# ─────────────────────────────────────────────────────────────────────────────

class FilterSelector:
    """
    Selects top-k features using statistical filter methods.
    Combines: Mutual Information + ANOVA F-statistic (averaged ranking).
    """

    def __init__(self, k: int = 20):
        self.k = k
        self.scores_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_names_: Optional[List[str]] = None
        self.importance_scores_: Optional[Dict[str, float]] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> "FilterSelector":
        n_features = X.shape[1]
        k = _validate_k(self.k, n_features)

        # Mutual Information
        mi = mutual_info_classif(X, y, random_state=42)
        mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-9)

        # ANOVA F-stat
        f_stat, _ = f_classif(X, y)
        f_stat = np.nan_to_num(f_stat, nan=0.0)
        f_norm = (f_stat - f_stat.min()) / (f_stat.max() - f_stat.min() + 1e-9)

        combined = 0.5 * mi_norm + 0.5 * f_norm
        self.scores_ = combined
        self.selected_indices_ = np.argsort(combined)[::-1][:k]

        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.selected_names_ = [names[i] for i in self.selected_indices_]
        self.importance_scores_ = {
            names[i]: float(combined[i]) for i in self.selected_indices_
        }
        logger.info(f"[Filter] Selected {k} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y, feature_names=None):
        return self.fit(X, y, feature_names).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper Methods (RFE)
# ─────────────────────────────────────────────────────────────────────────────

class WrapperSelector:
    """
    Recursive Feature Elimination (RFE) using Random Forest estimator.
    Iteratively removes the least important feature until k features remain.
    """

    def __init__(self, k: int = 20):
        self.k = k
        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_names_: Optional[List[str]] = None
        self.importance_scores_: Optional[Dict[str, float]] = None
        self.estimator_ = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> "WrapperSelector":
        n_features = X.shape[1]
        k = _validate_k(self.k, n_features)

        self.estimator_ = RandomForestClassifier(n_estimators=50, max_depth=10,
                                                   n_jobs=-1, random_state=42)
        rfe = RFE(self.estimator_, n_features_to_select=k, step=max(1, n_features // 10))
        rfe.fit(X, y)

        self.selected_indices_ = np.where(rfe.support_)[0]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.selected_names_ = [names[i] for i in self.selected_indices_]

        # Get importance scores
        self.estimator_.fit(X[:, self.selected_indices_], y)
        importances = self.estimator_.feature_importances_
        self.importance_scores_ = {
            names[i]: float(importances[j]) 
            for j, i in enumerate(self.selected_indices_)
        }
        logger.info(f"[Wrapper/RFE] Selected {k} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y, feature_names=None):
        return self.fit(X, y, feature_names).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Embedded Methods
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddedSelector:
    """
    Embedded feature selection using Random Forest feature importance.
    Selects top-k features by Gini importance.
    """

    def __init__(self, k: int = 20):
        self.k = k
        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_names_: Optional[List[str]] = None
        self.importance_scores_: Optional[Dict[str, float]] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> "EmbeddedSelector":
        n_features = X.shape[1]
        k = _validate_k(self.k, n_features)

        clf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                      n_jobs=-1, random_state=42)
        clf.fit(X, y)
        importances = clf.feature_importances_

        self.selected_indices_ = np.argsort(importances)[::-1][:k]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.selected_names_ = [names[i] for i in self.selected_indices_]
        self.importance_scores_ = {
            names[i]: float(importances[i]) for i in self.selected_indices_
        }
        logger.info(f"[Embedded/RF] Selected {k} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y, feature_names=None):
        return self.fit(X, y, feature_names).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Particle Swarm Optimization (PSO)
# ─────────────────────────────────────────────────────────────────────────────

class PSOSelector:
    """
    Particle Swarm Optimization for feature selection.
    Each particle is a binary vector representing feature selection.
    Fitness = cross-validated RF accuracy on selected features.
    """

    def __init__(self, k: int = 20, n_particles: int = 30, n_iter: int = 20,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.k = k
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_names_: Optional[List[str]] = None
        self.importance_scores_: Optional[Dict[str, float]] = None

    def _binary_to_features(self, mask: np.ndarray, n_features: int) -> np.ndarray:
        """Convert real-valued velocity to binary feature mask."""
        thresh = np.random.uniform(0.4, 0.6)  # Probabilistic binarization
        return (mask > thresh).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "PSOSelector":
        n_features = X.shape[1]
        k = _validate_k(self.k, n_features)

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y

        # Initialize particles
        particles = np.random.rand(self.n_particles, n_features)
        velocities = np.random.uniform(-1, 1, (self.n_particles, n_features))
        fitness = np.zeros(self.n_particles)

        # Evaluate initial population
        for i in range(self.n_particles):
            mask = self._binary_to_features(particles[i], n_features)
            fitness[i] = _score_subset(X_train, y_train, X_val, y_val, mask)

        pbest = particles.copy()
        pbest_fitness = fitness.copy()
        gbest_idx = np.argmax(fitness)
        gbest = particles[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]

        # PSO iterations
        for iteration in range(self.n_iter):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (pbest[i] - particles[i]) +
                    self.c2 * r2 * (gbest - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], 0, 1)

                mask = self._binary_to_features(particles[i], n_features)
                fitness[i] = _score_subset(X_train, y_train, X_val, y_val, mask)

                if fitness[i] > pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = fitness[i]

                    if fitness[i] > gbest_fitness:
                        gbest = particles[i].copy()
                        gbest_fitness = fitness[i]

            logger.info(f"[PSO] Iteration {iteration + 1}/{self.n_iter}, "
                       f"Best fitness: {gbest_fitness:.4f}")

        # Convert best solution to feature indices
        gbest_mask = self._binary_to_features(gbest, n_features)
        selected_count = gbest_mask.sum()
        if selected_count < k:
            top_k_idx = np.argsort(gbest)[::-1][:k]
            gbest_mask = np.zeros(n_features, dtype=int)
            gbest_mask[top_k_idx] = 1

        self.selected_indices_ = np.where(gbest_mask)[0]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.selected_names_ = [names[i] for i in self.selected_indices_]
        self.importance_scores_ = {
            names[i]: float(gbest[i]) for i in self.selected_indices_
        }
        logger.info(f"[PSO] Selected {len(self.selected_indices_)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y, feature_names=None, X_val=None, y_val=None):
        return self.fit(X, y, feature_names, X_val, y_val).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Ant Colony Optimization (ACO)
# ─────────────────────────────────────────────────────────────────────────────

class ACOSelector:
    """
    Ant Colony Optimization for feature selection.
    Pheromone levels guide feature selection probability.
    Fitness = RF accuracy on selected features.
    """

    def __init__(self, k: int = 20, n_ants: int = 30, n_iter: int = 20,
                 alpha: float = 1.0, beta: float = 2.0,
                 rho: float = 0.1, tau0: float = 0.5):
        self.k = k
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha  # pheromone importance
        self.beta = beta  # heuristic importance
        self.rho = rho  # evaporation rate
        self.tau0 = tau0  # initial pheromone
        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_names_: Optional[List[str]] = None
        self.importance_scores_: Optional[Dict[str, float]] = None

    def _select_features_probabilistic(self, tau: np.ndarray,
                                       eta: np.ndarray) -> np.ndarray:
        """Select features probabilistically based on pheromone & heuristic."""
        mask = np.zeros_like(tau, dtype=int)
        for i in range(len(tau)):
            prob = (tau[i] ** self.alpha) * (eta[i] ** self.beta)
            if prob > np.random.rand():
                mask[i] = 1
        return mask

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "ACOSelector":
        n_features = X.shape[1]
        k = _validate_k(self.k, n_features)

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y

        # Initialize pheromone and heuristic
        tau = np.full(n_features, self.tau0)  # Pheromone
        clf = RandomForestClassifier(n_estimators=30, max_depth=6,
                                      n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        eta = clf.feature_importances_  # Heuristic (feature importance)
        eta = (eta - eta.min()) / (eta.max() - eta.min() + 1e-9)

        best_mask = None
        best_fitness = 0.0

        # ACO iterations
        for iteration in range(self.n_iter):
            iteration_masks = []
            iteration_fitness = []

            # Ants construct solutions
            for ant in range(self.n_ants):
                mask = self._select_features_probabilistic(tau, eta)
                if mask.sum() == 0:
                    mask[np.argmax(eta)] = 1

                fitness = _score_subset(X_train, y_train, X_val, y_val, mask)
                iteration_masks.append(mask)
                iteration_fitness.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_mask = mask.copy()

            # Pheromone evaporation
            tau = tau * (1 - self.rho)

            # Pheromone deposit from best ants
            best_ants_idx = np.argsort(iteration_fitness)[-max(1, self.n_ants // 5):]
            for idx in best_ants_idx:
                tau += self.rho * iteration_fitness[idx] * iteration_masks[idx]

            logger.info(f"[ACO] Iteration {iteration + 1}/{self.n_iter}, "
                       f"Best fitness: {best_fitness:.4f}")

        # Ensure at least k features selected
        if best_mask.sum() < k:
            top_k_idx = np.argsort(eta)[::-1][:k]
            best_mask = np.zeros(n_features, dtype=int)
            best_mask[top_k_idx] = 1

        self.selected_indices_ = np.where(best_mask)[0]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.selected_names_ = [names[i] for i in self.selected_indices_]
        self.importance_scores_ = {
            names[i]: float(eta[i]) for i in self.selected_indices_
        }
        logger.info(f"[ACO] Selected {len(self.selected_indices_)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y, feature_names=None, X_val=None, y_val=None):
        return self.fit(X, y, feature_names, X_val, y_val).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_selector(method: str, k: int = 20, **kwargs):
    """
    Factory for feature selectors.
    method ∈ {"filter", "wrapper", "embedded", "pso", "aco"}
    """
    method = method.lower()
    if method == "filter":
        return FilterSelector(k=k)
    elif method == "wrapper":
        return WrapperSelector(k=k)
    elif method == "embedded":
        return EmbeddedSelector(k=k)
    elif method == "pso":
        return PSOSelector(k=k, **kwargs)
    elif method == "aco":
        return ACOSelector(k=k, **kwargs)
    else:
        raise ValueError(f"Unknown feature selection method: {method}. "
                         f"Choose from: filter, wrapper, embedded, pso, aco")