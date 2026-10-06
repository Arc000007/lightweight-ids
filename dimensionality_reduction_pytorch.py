"""
PyTorch-based Dimensionality Reduction.
- PCA (sklearn-based, unchanged)
- Autoencoder (PyTorch implementation for speed)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# PCA Dimensionality Reduction
# ─────────────────────────────────────────────────────────────────────────────

class PCAReducer:
    """PCA-based dimensionality reduction."""

    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.pca = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """Fit PCA."""
        self.pca = PCA(n_components=min(self.n_components, X.shape[1]))
        self.pca.fit(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        logger.info(f"[PCA] Explained variance: "
                   f"{self.explained_variance_ratio_.sum():.2%}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)

    def get_variance_explained(self) -> float:
        """Get total explained variance ratio."""
        return float(self.explained_variance_ratio_.sum())


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class AutoencoderNet(nn.Module):
    """PyTorch autoencoder for dimensionality reduction."""

    def __init__(self, input_dim: int, encoding_dim: int):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Get encoded representation."""
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder-based Dimensionality Reduction
# ─────────────────────────────────────────────────────────────────────────────

class AutoencoderReducer:
    """PyTorch-based autoencoder for dimensionality reduction."""

    def __init__(self, input_dim: int, encoding_dim: int = 20,
                 epochs: int = 50, batch_size: int = 32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional[AutoencoderNet] = None
        self.device = DEVICE
        self.history = None

    def fit(self, X: np.ndarray, validation_split: float = 0.1) -> "AutoencoderReducer":
        """Train autoencoder."""
        # Create model
        self.model = AutoencoderNet(self.input_dim, self.encoding_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        # Split
        split_idx = int(len(X) * (1 - validation_split))
        X_train = X_tensor[:split_idx]
        X_val = X_tensor[split_idx:]
        
        losses = []
        logger.info(f"[Autoencoder] Training for {self.epochs} epochs on {len(X)} samples")
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Training
            indices = np.random.permutation(len(X_train))
            train_loss = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                batch = X_train[batch_idx]
                
                optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= max(1, len(X_train) // self.batch_size)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val)
                val_loss = criterion(val_output, X_val).item()
            
            losses.append(train_loss)
            
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.debug(f"  Epoch {epoch+1}/{self.epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        self.history = {"loss": losses}
        logger.info(f"[Autoencoder] Training complete. Final loss: {losses[-1]:.4f}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce dimensions using encoder."""
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            encoded = self.model.encode(X_tensor)
        
        return encoded.cpu().numpy()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)

    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Get per-sample reconstruction error."""
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            X_reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - X_reconstructed) ** 2, dim=1)
        
        return errors.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality Reduction Factory
# ─────────────────────────────────────────────────────────────────────────────

class DimensionalityReducer:
    """Factory for dimensionality reduction."""

    @staticmethod
    def create_reducer(method: str, input_dim: int, encoding_dim: int = 20):
        """Create a reducer based on method."""
        if method.lower() == 'none':
            return None
        elif method.lower() == 'pca':
            return PCAReducer(n_components=encoding_dim)
        elif method.lower() == 'autoencoder':
            return AutoencoderReducer(
                input_dim=input_dim,
                encoding_dim=encoding_dim,
                epochs=50,
                batch_size=32
            )
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
