"""
Deep learning models for IDS (PyTorch backend).
- Attention-Based Autoencoder
- CNN + BiLSTM Hybrid
- Temporal Convolutional Network (TCN)
"""

import math
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Utilities
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm1d(nn.Module):
    """LayerNorm that works on (B, T, C) or (B, C) tensors."""

    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


# ─────────────────────────────────────────────────────────────────────────────
# Attention Mechanism
# ─────────────────────────────────────────────────────────────────────────────

class AttentionBlock(nn.Module):
    """
    Multi-head self-attention + residual + LayerNorm.
    Expects input shape  (B, seq_len, embed_dim).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        # embed_dim must be divisible by num_heads
        if embed_dim % num_heads != 0:
            # find the largest valid divisor ≤ num_heads
            num_heads = max(h for h in range(1, num_heads + 1) if embed_dim % h == 0)
            logger.warning(f"Adjusted num_heads to {num_heads} for embed_dim={embed_dim}")

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,   # (B, T, C) convention – matches Keras default
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


# ─────────────────────────────────────────────────────────────────────────────
# Attention-Based Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class AttentionAutoencoder(nn.Module):
    """
    Autoencoder with self-attention for anomaly detection.
    Input / output shape: (B, input_dim)  — flat feature vectors.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 16, num_heads: int = 4):
        super().__init__()

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc_fc1    = nn.Linear(input_dim, 128)
        self.enc_drop1  = nn.Dropout(0.2)
        self.enc_attn   = AttentionBlock(embed_dim=128, num_heads=num_heads)
        self.enc_fc2    = nn.Linear(128, 64)
        self.enc_drop2  = nn.Dropout(0.2)
        self.enc_fc3    = nn.Linear(64, 32)
        self.bottleneck = nn.Linear(32, encoding_dim)

        # ── Decoder ────────────────────────────────────────────────────────
        self.dec_fc1   = nn.Linear(encoding_dim, 32)
        self.dec_drop1 = nn.Dropout(0.2)
        self.dec_fc2   = nn.Linear(32, 64)
        self.dec_fc3   = nn.Linear(64, 128)
        self.dec_attn  = AttentionBlock(embed_dim=128, num_heads=num_heads)
        self.dec_fc4   = nn.Linear(128, 128)
        self.dec_drop2 = nn.Dropout(0.2)
        self.output    = nn.Linear(128, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent (bottleneck) representation."""
        x = F.relu(self.enc_fc1(x))
        x = self.enc_drop1(x)
        x = x.unsqueeze(1)                     # (B, 1, 128) — seq_len=1 for attention
        x = self.enc_attn(x).squeeze(1)        # (B, 128)
        x = F.relu(self.enc_fc2(x))
        x = self.enc_drop2(x)
        x = F.relu(self.enc_fc3(x))
        return F.relu(self.bottleneck(x))       # (B, encoding_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dec_fc1(z))
        x = self.dec_drop1(x)
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = x.unsqueeze(1)                     # (B, 1, 128)
        x = self.dec_attn(x).squeeze(1)        # (B, 128)
        x = F.relu(self.dec_fc4(x))
        x = self.dec_drop2(x)
        return torch.sigmoid(self.output(x))   # (B, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstruction — same shape as input."""
        return self.decode(self.encode(x))


# ─────────────────────────────────────────────────────────────────────────────
# CNN + BiLSTM Hybrid
# ─────────────────────────────────────────────────────────────────────────────

class CNNBiLSTM(nn.Module):
    """
    CNN + Bidirectional LSTM for sequence classification.
    Input shape : (B, seq_length, features_per_step)
    Output shape: (B, num_classes)
    """

    def __init__(self, input_dim: int, seq_length: int = 30, num_classes: int = 2):
        super().__init__()

        # input_dim here is features_per_step (not the flat feature count)
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(kernel_size=2)

        # After MaxPool1d(2) the sequence halves: seq_length // 2
        lstm_input_size = 64
        self.bilstm1 = nn.LSTM(lstm_input_size, 64, batch_first=True, bidirectional=True)
        self.drop1   = nn.Dropout(0.3)
        # BiLSTM doubles the hidden size: 64 * 2 = 128
        self.bilstm2 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.drop2   = nn.Dropout(0.3)

        # Classification head  (BiLSTM2 output = 32*2 = 64)
        self.fc1    = nn.Linear(64, 32)
        self.drop3  = nn.Dropout(0.2)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  permute for Conv1d: (B, C, T)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)                        # (B, 64, T//2)

        x = x.permute(0, 2, 1)                 # back to (B, T//2, 64) for LSTM
        x, _ = self.bilstm1(x)                 # (B, T//2, 128)
        x = self.drop1(x)
        x, _ = self.bilstm2(x)                 # (B, T//2, 64)
        x = self.drop2(x)

        x = x[:, -1, :]                        # take last time-step (B, 64)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        return self.output(x)                  # (B, num_classes) — raw logits


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Convolutional Network (TCN)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Dilated causal residual block for TCN.
    Input / output shape: (B, channels, T)  — channels-first, matching Conv1d.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        # Causal padding so no future information leaks
        self.pad = (kernel_size - 1) * dilation

        self.conv1   = nn.Conv1d(in_channels,  out_channels, kernel_size,
                                 dilation=dilation, padding=0)
        self.conv2   = nn.Conv1d(out_channels, out_channels, kernel_size,
                                 dilation=dilation, padding=0)
        self.dropout = nn.Dropout(0.2)
        self.norm1   = nn.LayerNorm(out_channels)  # applied after permute
        self.norm2   = nn.LayerNorm(out_channels)

        # 1×1 projection for residual when channel dims differ
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = self.residual_proj(x)

        # First dilated conv + causal padding
        y = F.pad(x, (self.pad, 0))
        y = F.relu(self.conv1(y))
        y = self.dropout(y)
        y = y.permute(0, 2, 1)       # (B, T, C) for LayerNorm
        y = self.norm1(y).permute(0, 2, 1)   # back to (B, C, T)

        # Second dilated conv + causal padding
        y = F.pad(y, (self.pad, 0))
        y = self.conv2(y)
        y = self.dropout(y)
        y = y.permute(0, 2, 1)
        y = self.norm2(y).permute(0, 2, 1)

        return F.relu(y + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for IDS classification.
    Input shape : (B, seq_length, features_per_step)
    Output shape: (B, num_classes)
    """

    def __init__(self, input_dim: int, seq_length: int = 30,
                 num_filters: int = 64, num_classes: int = 2):
        super().__init__()

        dilation_rates = [1, 2, 4, 8]

        # Build TCN backbone
        blocks = []
        in_ch  = input_dim
        for d in dilation_rates:
            blocks.append(ResidualBlock(in_ch, num_filters, dilation=d))
            in_ch = num_filters
        self.tcn_blocks = nn.Sequential(*blocks)

        # Classification head
        self.fc1    = nn.Linear(num_filters, 64)
        self.drop1  = nn.Dropout(0.3)
        self.fc2    = nn.Linear(64, 32)
        self.drop2  = nn.Dropout(0.2)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.tcn_blocks(x)         # (B, num_filters, T)
        x = x.mean(dim=2)              # global average pooling → (B, num_filters)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.output(x)          # (B, num_classes) — raw logits


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

class ModelFactory:
    """Factory for creating PyTorch IDS models."""

    @staticmethod
    def create_model(model_type: str, input_dim: int, seq_length: int = 30,
                     encoding_dim: int = 16, num_classes: int = 2,
                     num_filters: int = 64) -> nn.Module:
        """
        Create and return an nn.Module for the given model_type.

        For sequence models (cnn_bilstm / tcn) the caller must reshape its
        flat feature vectors from (N, input_dim) to
        (N, seq_length, input_dim // seq_length) before passing to the model.

        Parameters
        ----------
        model_type   : 'attention_autoencoder' | 'cnn_bilstm' | 'tcn'
        input_dim    : total number of features (flat)
        seq_length   : time-steps for sequence models
        encoding_dim : bottleneck size for the autoencoder
        num_classes  : output classes for classifiers
        num_filters  : TCN channel width
        """
        if model_type == 'attention_autoencoder':
            model = AttentionAutoencoder(
                input_dim=input_dim,
                encoding_dim=encoding_dim,
            )
            logger.info(
                f"Built AttentionAutoencoder "
                f"(input={input_dim}, bottleneck={encoding_dim})"
            )
            return model

        elif model_type == 'cnn_bilstm':
            features_per_step = input_dim // seq_length
            if input_dim % seq_length != 0:
                raise ValueError(
                    f"input_dim ({input_dim}) must be divisible by "
                    f"seq_length ({seq_length}) for CNN+BiLSTM"
                )
            model = CNNBiLSTM(
                input_dim=features_per_step,
                seq_length=seq_length,
                num_classes=num_classes,
            )
            logger.info(
                f"Built CNNBiLSTM "
                f"(seq={seq_length}, feat/step={features_per_step}, classes={num_classes})"
            )
            return model

        elif model_type == 'tcn':
            features_per_step = input_dim // seq_length
            if input_dim % seq_length != 0:
                raise ValueError(
                    f"input_dim ({input_dim}) must be divisible by "
                    f"seq_length ({seq_length}) for TCN"
                )
            model = TCN(
                input_dim=features_per_step,
                seq_length=seq_length,
                num_filters=num_filters,
                num_classes=num_classes,
            )
            logger.info(
                f"Built TCN "
                f"(seq={seq_length}, feat/step={features_per_step}, "
                f"filters={num_filters}, classes={num_classes})"
            )
            return model

        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                "Choose from: 'attention_autoencoder', 'cnn_bilstm', 'tcn'."
            )

    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Return approximate model size in MB (float32 weights only)."""
        total_params = sum(p.numel() for p in model.parameters())
        return (total_params * 4) / (1024 ** 2)

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def model_summary(model: nn.Module, input_dim: int,
                      seq_length: Optional[int] = None, device: str = 'cpu'):
        """
        Print a concise parameter summary (no torchsummary dependency).
        """
        total   = ModelFactory.count_parameters(model)
        size_mb = ModelFactory.get_model_size(model)
        print(f"\n{'─'*45}")
        print(f"  Model : {model.__class__.__name__}")
        print(f"  Params: {total:,}")
        print(f"  Size  : {size_mb:.3f} MB")
        print(f"{'─'*45}\n")