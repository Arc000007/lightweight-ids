"""
PyTorch deep learning models for IDS.
- Attention-Based Autoencoder
- CNN + BiLSTM Hybrid
- Temporal Convolutional Network (TCN)
Faster training than TensorFlow with GPU support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Attention Mechanism
# ─────────────────────────────────────────────────────────────────────────────

class AttentionLayer(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] → [B, 1, D]
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


# ─────────────────────────────────────────────────────────────────────────────
# Attention-Based Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class AttentionAutoencoder(nn.Module):
    """
    Autoencoder with attention mechanism for anomaly detection.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 16, num_heads: int = 4):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        self.attention = AttentionLayer(32, num_heads)
        
        # Bottleneck
        self.bottleneck = nn.Linear(32, encoding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        attended = self.attention(encoded)
        z = self.bottleneck(attended)
        
        # Decode
        decoded = self.decoder(z)
        return decoded, z


# ─────────────────────────────────────────────────────────────────────────────
# CNN + BiLSTM Hybrid
# ─────────────────────────────────────────────────────────────────────────────

class CNNBiLSTM(nn.Module):
    """
    CNN + BiLSTM hybrid model for sequence-based anomaly detection.
    Extracts spatial features via CNN, temporal dependencies via BiLSTM.
    """

    def __init__(self, input_dim: int, seq_length: int = 30, num_classes: int = 2):
        super().__init__()
        
        # CNN feature extraction
        self.conv1 = nn.Conv1d(seq_length, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        
        # BiLSTM
        self.lstm = nn.LSTM(input_dim // 2, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        
        # Classifier
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # CNN: [B, seq_len, input_dim] → [B, 64, input_dim//2]
        x_cnn = F.relu(self.bn1(self.conv1(x)))
        x_cnn = F.relu(self.bn2(self.conv2(x_cnn)))
        x_cnn = self.pool(x_cnn)
        
        # BiLSTM: [B, seq_len, input_dim] → [B, 128]
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last timestep
        lstm_out = self.dropout(lstm_out)
        
        # Classifier
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Convolutional Network (TCN)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual block for TCN."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=dilation, padding_mode='same'
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=dilation, padding_mode='same'
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + residual
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for IDS.
    Uses dilated convolutions to capture long-range temporal dependencies.
    """

    def __init__(self, input_dim: int, seq_length: int = 30, num_filters: int = 64, num_classes: int = 2):
        super().__init__()
        
        # TCN backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(num_filters, dilation=1),
            ResidualBlock(num_filters, dilation=2),
            ResidualBlock(num_filters, dilation=4),
            ResidualBlock(num_filters, dilation=8),
        ])
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, num_filters)
        
        # Classifier head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Project input: [B, seq_len, input_dim] → [B, seq_len, num_filters]
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # [B, num_filters, seq_len]
        
        # TCN blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling: [B, num_filters, seq_len] → [B, num_filters]
        x = self.pool(x).squeeze(-1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

class ModelFactory:
    """Factory for creating and compiling PyTorch models."""

    @staticmethod
    def create_model(model_type: str, input_dim: int, seq_length: int = 30,
                     encoding_dim: int = 16, num_classes: int = 2):
        """Create a PyTorch model based on type."""
        if model_type == 'attention_autoencoder':
            return AttentionAutoencoder(input_dim, encoding_dim)
        elif model_type == 'cnn_bilstm':
            return CNNBiLSTM(input_dim, seq_length, num_classes)
        elif model_type == 'tcn':
            return TCN(input_dim, seq_length, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Get model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per float32
        return size_mb

    @staticmethod
    def get_device():
        """Get the appropriate device (cuda or cpu)."""
        return DEVICE

    @staticmethod
    def move_to_device(model: nn.Module) -> nn.Module:
        """Move model to the appropriate device."""
        return model.to(DEVICE)
