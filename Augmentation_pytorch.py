"""
WGAN-GP-based data augmentation using PyTorch.
Generates synthetic attack/benign samples for class balancing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Generator and Critic Networks
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """Generator network for WGAN-GP."""

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
        )

    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    """Critic network for WGAN-GP (no final sigmoid for Wasserstein loss)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # Single scalar output, no sigmoid
        )

    def forward(self, x):
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# WGAN-GP Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                             device: torch.device, lambda_gp: float = 10) -> torch.Tensor:
    """Compute gradient penalty for Lipschitz constraint."""
    batch_size = real.size(0)
    
    # Random weights
    alpha = torch.rand(batch_size, 1, device=device)
    
    # Interpolation
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    # Critic on interpolated samples
    d_interpolates = critic(interpolates)
    fake_labels = torch.ones(batch_size, 1, device=device, requires_grad=True)
    
    # Backward to compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_labels,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


# ─────────────────────────────────────────────────────────────────────────────
# WGAN Augmenter
# ─────────────────────────────────────────────────────────────────────────────

class WGANAugmenter:
    """WGAN-GP-based augmenter for synthetic data generation."""

    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        self.generators = {}
        self.device = DEVICE

    def _train_class(self, X_class: np.ndarray, class_label: int,
                     epochs: int = 10, batch_size: int = 32) -> Tuple[Generator, Critic]:
        """Train GAN on a single class."""
        
        output_dim = X_class.shape[1]
        generator = Generator(self.latent_dim, output_dim).to(self.device)
        critic = Critic(output_dim).to(self.device)
        
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        c_optimizer = optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Convert to tensor
        X_tensor = torch.from_numpy(X_class.astype(np.float32)).to(self.device)
        
        logger.info(f"Training WGAN for class {class_label}: {len(X_class)} samples, {output_dim} features")
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(len(X_tensor))
            
            for i in range(0, len(X_tensor), batch_size):
                batch_idx = indices[i:i+batch_size]
                real_batch = X_tensor[batch_idx]
                
                if len(real_batch) < batch_size:
                    continue
                
                # Train critic
                c_optimizer.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake = generator(z).detach()
                
                c_real = critic(real_batch)
                c_fake = critic(fake)
                
                # Wasserstein loss: max E[C(real)] - E[C(fake)]
                critic_loss = -c_real.mean() + c_fake.mean()
                
                # Gradient penalty
                gp = compute_gradient_penalty(critic, real_batch, fake, self.device)
                critic_loss += gp
                
                critic_loss.backward()
                c_optimizer.step()
                
                # Train generator
                g_optimizer.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake = generator(z)
                c_fake = critic(fake)
                
                # Generator loss: max E[C(fake)]
                g_loss = -c_fake.mean()
                
                g_loss.backward()
                g_optimizer.step()
            
            if (epoch + 1) % max(1, epochs // 3) == 0:
                logger.debug(f"  Epoch {epoch+1}/{epochs} - G_loss: {g_loss.item():.4f}, C_loss: {critic_loss.item():.4f}")
        
        logger.info(f"WGAN training complete for class {class_label}")
        return generator, critic

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        """Train GANs on both classes."""
        for class_label in [0, 1]:
            mask = y == class_label
            X_class = X[mask]
            self.generators[class_label] = self._train_class(
                X_class, class_label, epochs, batch_size
            )

    def generate(self, class_label: int, n_samples: int) -> np.ndarray:
        """Generate synthetic samples for a specific class."""
        if class_label not in self.generators:
            raise ValueError(f"No generator trained for class {class_label}")
        
        generator, _ = self.generators[class_label]
        generator.eval()
        
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            synthetic = generator(z).cpu().numpy()
        
        return synthetic

    def generate_balanced(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate equal samples for both classes."""
        X_benign = self.generate(0, n_samples // 2)
        X_attack = self.generate(1, n_samples // 2)
        
        X = np.vstack([X_benign, X_attack])
        y = np.array([0] * len(X_benign) + [1] * len(X_attack))
        
        # Shuffle
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]
