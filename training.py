"""
Training pipeline for IDS models.
Orchestrates data preprocessing, feature selection, model training, and evaluation.
Converted from TensorFlow/Keras to PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
import os
from typing import Dict, Optional, Tuple
from pathlib import Path

from Preprocessing import DataPreprocessor
from Feature_selection import FilterSelector, WrapperSelector, EmbeddedSelector, PSOSelector, ACOSelector
from dimensionality_reduction_pytorch import DimensionalityReducer
from Augmentation_pytorch import WGANAugmenter
from models import ModelFactory
from thresholding import ThresholdFactory
from evaluation import IDSEvaluator, PerformanceAnalyzer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Device Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Training Utilities
# ─────────────────────────────────────────────────────────────────────────────

def numpy_to_tensor(*arrays, dtype=torch.float32):
    """Convert numpy arrays to PyTorch tensors on the correct device."""
    return [torch.tensor(a, dtype=dtype).to(DEVICE) for a in arrays]


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int,
                    shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from numpy arrays."""
    X_t, y_t = numpy_to_tensor(X, y)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module,
                    is_autoencoder: bool = False) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()

        output = model(X_batch)

        if is_autoencoder:
            # Reconstruction loss — target is the input itself
            loss = criterion(output, X_batch)
        else:
            # Classification loss
            if output.shape[-1] > 1:
                loss = criterion(output, y_batch.long())
            else:
                loss = criterion(output.squeeze(), y_batch.float())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_one_epoch(model: nn.Module, loader: DataLoader,
                       criterion: nn.Module,
                       is_autoencoder: bool = False) -> float:
    """Evaluate on a validation set. Returns mean loss."""
    model.eval()
    total_loss = 0.0

    for batch in loader:
        X_batch, y_batch = batch
        output = model(X_batch)

        if is_autoencoder:
            loss = criterion(output, X_batch)
        else:
            if output.shape[-1] > 1:
                loss = criterion(output, y_batch.long())
            else:
                loss = criterion(output.squeeze(), y_batch.float())

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Run inference and return numpy predictions."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    outputs = []
    for (batch,) in loader:
        out = model(batch)
        outputs.append(out.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def get_model_size_mb(model: nn.Module) -> float:
    """Return approximate model size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    # Each parameter is a float32 (4 bytes)
    return (total_params * 4) / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# IDS Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class IDSTrainingPipeline:
    """End-to-end IDS training pipeline (PyTorch backend)."""

    def __init__(self, config: Dict):
        """
        Initialize pipeline with configuration.

        config: Dict containing:
          - dataset_name: str
          - data_path: str
          - feature_selection_method: str
          - feature_k: int
          - augmentation_enabled: bool
          - dimensionality_reduction: str
          - reduction_dim: int
          - model_type: str          ('attention_autoencoder' | 'cnn_bilstm' | 'tcn')
          - hyperparameter_optimization: str
          - threshold_method: str    (for autoencoder only)
          - epochs: int
          - batch_size: int
          - learning_rate: float
        """
        self.config = config
        self.preprocessor = None
        self.feature_selector = None
        self.reducer = None
        self.model: Optional[nn.Module] = None
        self.threshold = None
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        self.evaluation_results = None
        self.performance_metrics = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _create_feature_selector(self):
        method = self.config.get('feature_selection_method', 'filter').lower()
        k = self.config.get('feature_k', 20)

        selectors = {
            'filter':   lambda: FilterSelector(k=k),
            'wrapper':  lambda: WrapperSelector(k=k),
            'embedded': lambda: EmbeddedSelector(k=k),
            'pso':      lambda: PSOSelector(k=k, n_particles=30, n_iter=20),
            'aco':      lambda: ACOSelector(k=k, n_ants=30, n_iter=20),
        }

        creator = selectors.get(method)
        if creator is None:
            logger.warning(f"Unknown feature selection method '{method}', using filter")
            creator = selectors['filter']
        return creator()

    def _is_autoencoder(self) -> bool:
        return self.config.get('model_type', '') == 'attention_autoencoder'

    # ── Pipeline Stages ──────────────────────────────────────────────────────

    def load_data(self, max_rows: Optional[int] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess data."""
        logger.info("Loading and preprocessing data...")
        
        # Auto-detect dataset name if not explicitly provided
        dataset_name = self.config.get('dataset_name')
        if not dataset_name or dataset_name == 'Auto-detect':
            dataset_name = self._detect_dataset_name(self.config['data_path'])
            if dataset_name:
                logger.info(f"Auto-detected dataset: {dataset_name}")
                self.config['dataset_name'] = dataset_name

        self.preprocessor = DataPreprocessor(
            dataset_name=dataset_name or 'CICIDS2017',
            data_path=self.config['data_path'],
            test_size=0.2,
            val_size=0.1,
        )

        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.preprocess(
            max_rows=max_rows
        )

        logger.info(
            f"Data loaded — Train: {X_train.shape}, "
            f"Val: {X_val.shape}, Test: {X_test.shape}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _detect_dataset_name(self, data_path: str) -> Optional[str]:
        """Auto-detect dataset name based on directory structure."""
        import os
        path_lower = str(data_path).lower()
        dir_name = os.path.basename(path_lower)
        
        # Check known dataset patterns
        if 'ciciot' in path_lower or 'ciciot23' in dir_name:
            return 'CICIoT2023'
        elif 'ton' in path_lower or 'ton_iot' in dir_name:
            return 'TON_IoT'
        elif 'cicids2017' in path_lower or 'cic-ids-2017' in dir_name or 'cicids' in dir_name:
            return 'CICIDS2017'
        elif 'cse' in path_lower or 'cic-ids2018' in dir_name:
            return 'CSE-CIC-IDS2018'
        elif 'edge' in path_lower or 'iiot' in dir_name:
            return 'Edge-IIoTset'
        elif 'iot-23' in path_lower or 'iot23' in dir_name:
            return 'IoT-23'
        
        return None

    def select_features(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        feature_names: Optional[list] = None
                        ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Select features using the configured method."""
        logger.info(
            f"Selecting features using '{self.config['feature_selection_method']}'..."
        )

        self.feature_selector = self._create_feature_selector()

        if hasattr(self.feature_selector, 'fit'):
            if self.feature_selector.__class__.__name__ in ('PSOSelector', 'ACOSelector'):
                self.feature_selector.fit(X_train, y_train, feature_names, X_val, y_val)
            else:
                self.feature_selector.fit(X_train, y_train, feature_names)

        X_train_selected = self.feature_selector.transform(X_train)
        X_val_selected = self.feature_selector.transform(X_val)

        importance_scores = getattr(self.feature_selector, 'importance_scores_', {}) or {}
        logger.info(f"Selected {X_train_selected.shape[1]} features")

        return X_train_selected, X_val_selected, importance_scores

    def reduce_dimensions(self, X_train: np.ndarray, X_val: np.ndarray,
                          X_test: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reduce dimensions if configured."""
        method = self.config.get('dimensionality_reduction', 'none').lower()

        if method == 'none':
            return X_train, X_val, X_test

        logger.info(f"Reducing dimensions using '{method}'...")

        self.reducer = DimensionalityReducer.create_reducer(
            method,
            input_dim=X_train.shape[1],
            encoding_dim=self.config.get('reduction_dim', 20),
        )

        if self.reducer:
            self.reducer.fit(X_train)
            X_train = self.reducer.transform(X_train)
            X_val   = self.reducer.transform(X_val)
            X_test  = self.reducer.transform(X_test)
            logger.info(f"Reduced to {X_train.shape[1]} dimensions")

        return X_train, X_val, X_test

    def augment_data(self, X_train: np.ndarray,
                     y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment training data using GAN if enabled."""
        if not self.config.get('augmentation_enabled', False):
            return X_train, y_train

        logger.info("Augmenting data using WGAN-GP...")

        augmenter = WGANAugmenter(latent_dim=64)
        augmenter.train(X_train, y_train, epochs=3, batch_size=64)

        n_synthetic = min(X_train.shape[0] // 4, 1000)
        X_synthetic, y_synthetic = augmenter.generate_balanced(n_synthetic)

        X_train = np.vstack([X_train, X_synthetic])
        y_train = np.hstack([y_train, y_synthetic])

        logger.info(f"Data augmented: {X_train.shape}")
        return X_train, y_train

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> nn.Module:
        """Train the main model using PyTorch."""
        model_type = self.config.get('model_type', 'attention_autoencoder')
        input_dim  = X_train.shape[1]
        seq_length = self.config.get('seq_length', 30)
        epochs     = self.config.get('epochs', 50)
        batch_size = self.config.get('batch_size', 32)
        lr         = self.config.get('learning_rate', 1e-3)

        logger.info(f"Training model: '{model_type}' for {epochs} epochs...")

        is_ae = self._is_autoencoder()

        # Reshape for sequence models
        if not is_ae and model_type in ('cnn_bilstm', 'tcn'):
            features_per_step = input_dim // seq_length
            X_train = X_train.reshape(X_train.shape[0], seq_length, features_per_step)
            X_val   = X_val.reshape(X_val.shape[0],   seq_length, features_per_step)

        # Build model via your existing ModelFactory (must return nn.Module now)
        self.model = ModelFactory.create_model(model_type, input_dim, seq_length)
        self.model.to(DEVICE)

        # Loss function
        if is_ae:
            criterion = nn.MSELoss()
        else:
            n_classes = len(np.unique(y_train))
            criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
        )

        train_loader = make_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader   = make_dataloader(X_val,   y_val,   batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_state    = None

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                self.model, train_loader, optimizer, criterion, is_autoencoder=is_ae
            )
            val_loss = evaluate_one_epoch(
                self.model, val_loader, criterion, is_autoencoder=is_ae
            )
            scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:>3}/{epochs} — "
                    f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}"
                )

            # Keep best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        return self.model

    def setup_thresholding(self, X_train: np.ndarray,
                           y_train: np.ndarray) -> Optional[float]:
        """Setup thresholding for the autoencoder model."""
        if not self._is_autoencoder():
            return None

        method = self.config.get('threshold_method', 'statistical').lower()
        logger.info(f"Setting up thresholding: '{method}'...")

        X_pred = predict(self.model, X_train)
        reconstruction_errors = np.mean((X_train - X_pred) ** 2, axis=1)

        self.threshold = ThresholdFactory.create_threshold(method)
        threshold = (
            self.threshold.compute(reconstruction_errors, y_train)
            if method == 'pso'
            else self.threshold.compute(reconstruction_errors)
        )

        logger.info(f"Threshold set to: {threshold:.4f}")
        return threshold

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 model_save_path: Optional[str] = None) -> Dict:
        """Evaluate model on the test set."""
        logger.info("Evaluating model...")

        y_pred_proba = predict(self.model, X_test)

        if self._is_autoencoder():
            reconstruction_error = np.mean((X_test - y_pred_proba) ** 2, axis=1)
            y_pred = self.threshold.classify(reconstruction_error)
        else:
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        evaluator = IDSEvaluator()
        self.evaluation_results = evaluator.evaluate(y_test, y_pred, y_pred_proba)

        # Performance metrics
        model_size_mb = get_model_size_mb(self.model)   # ← replaces ModelFactory.get_model_size
        analyzer      = PerformanceAnalyzer()

        # Measure inference time
        start = time.time()
        _ = predict(self.model, X_test[:100])
        inference_time_ms = (time.time() - start) / 100 * 1000   # convert to ms

        n_features    = X_test.shape[1]
        total_features = (
            self.preprocessor.n_features if self.preprocessor else n_features
        )

        self.performance_metrics = analyzer.analyze(
            model_size_mb=model_size_mb,
            inference_time_ms=inference_time_ms,
            n_features_selected=n_features,
            total_features=total_features,
        )

        if model_save_path:
            self.save_model(model_save_path)

        logger.info("Evaluation complete")
        return {
            'evaluation': self.evaluation_results,
            'performance': self.performance_metrics,
        }

    def get_summary(self) -> Dict:
        """Return a JSON-serialisable training summary."""
        # Clean feature importance scores for JSON serialization (handle NaN)
        feature_importance = {}
        if self.feature_selector and hasattr(self.feature_selector, 'importance_scores_'):
            importance_scores = self.feature_selector.importance_scores_
            if importance_scores:
                for key, value in importance_scores.items():
                    if isinstance(value, float):
                        # Replace NaN with None (JSON null)
                        feature_importance[key] = float(value) if not np.isnan(value) else None
                    else:
                        feature_importance[key] = value
        
        return {
            'config': self.config,
            'evaluation': self.evaluation_results,
            'performance': self.performance_metrics,
            'feature_importance': feature_importance if feature_importance else None,
        }

    # ── Model persistence ────────────────────────────────────────────────────

    def save_model(self, path: str):
        """Save the trained model's state dict."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str, model_type: Optional[str] = None):
        """Load a saved model state dict."""
        checkpoint = torch.load(path, map_location=DEVICE)
        cfg        = checkpoint.get('config', self.config)
        mtype      = model_type or cfg.get('model_type', 'attention_autoencoder')

        self.model = ModelFactory.create_model(
            mtype,
            cfg.get('input_dim', 20),
            cfg.get('seq_length', 30),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(DEVICE)
        self.model.eval()
        logger.info(f"Model loaded from {path}")