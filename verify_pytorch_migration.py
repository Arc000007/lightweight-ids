#!/usr/bin/env python3
"""
Verification script for PyTorch migration.
Tests imports, model creation, and training pipeline.
"""

import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 70)
print("PyTorch IDS Migration Verification")
print("=" * 70)

# Test 1: Check PyTorch installation
print("\n[1] Checking PyTorch installation...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"  ✗ PyTorch import failed: {e}")
    sys.exit(1)

# Test 2: Test models import
print("\n[2] Testing models import...")
try:
    from models import ModelFactory, AttentionAutoencoder, CNNBiLSTM, TCN
    print(f"  ✓ models.py imports successful")
except ImportError as e:
    print(f"  ✗ models.py import failed: {e}")
    sys.exit(1)

# Test 3: Test dimensionality_reduction_pytorch import
print("\n[3] Testing dimensionality_reduction_pytorch import...")
try:
    from dimensionality_reduction_pytorch import DimensionalityReducer, AutoencoderReducer, PCAReducer
    print(f"  ✓ dimensionality_reduction_pytorch.py imports successful")
except ImportError as e:
    print(f"  ✗ dimensionality_reduction_pytorch.py import failed: {e}")
    sys.exit(1)

# Test 4: Test Augmentation_pytorch import
print("\n[4] Testing Augmentation_pytorch import...")
try:
    from Augmentation_pytorch import WGANAugmenter, Generator, Critic
    print(f"  ✓ Augmentation_pytorch.py imports successful")
except ImportError as e:
    print(f"  ✗ Augmentation_pytorch.py import failed: {e}")
    sys.exit(1)

# Test 5: Test model creation
print("\n[5] Testing model creation...")
try:
    input_dim = 40
    seq_length = 30
    
    # Attention Autoencoder
    ae = ModelFactory.create_model('attention_autoencoder', input_dim)
    print(f"  ✓ Attention Autoencoder created ({ModelFactory.get_model_size(ae):.2f} MB)")
    
    # CNN + BiLSTM (input_dim must be divisible by seq_length)
    # Use 60 features (divisible by 30) for sequence models
    input_dim_seq = 60
    cnn_lstm = ModelFactory.create_model('cnn_bilstm', input_dim_seq, seq_length=30)
    print(f"  ✓ CNN+BiLSTM created ({ModelFactory.get_model_size(cnn_lstm):.2f} MB)")
    
    # TCN
    tcn = ModelFactory.create_model('tcn', input_dim_seq, seq_length=30)
    print(f"  ✓ TCN created ({ModelFactory.get_model_size(tcn):.2f} MB)")
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test dimensionality reducer creation
print("\n[6] Testing dimensionality reducer creation...")
try:
    # PCA
    pca = DimensionalityReducer.create_reducer('pca', input_dim=40, encoding_dim=20)
    print(f"  ✓ PCA reducer created")
    
    # Autoencoder
    ae_reducer = DimensionalityReducer.create_reducer('autoencoder', input_dim=40, encoding_dim=20)
    print(f"  ✓ Autoencoder reducer created")
except Exception as e:
    print(f"  ✗ Dimensionality reducer creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test forward pass
print("\n[7] Testing forward pass...")
try:
    X = np.random.randn(10, 40).astype(np.float32)
    X_tensor = torch.from_numpy(X).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Autoencoder forward
    ae = ModelFactory.create_model('attention_autoencoder', input_dim=40)
    ae.eval()
    with torch.no_grad():
        out_ae = ae(X_tensor)
    print(f"  ✓ Attention Autoencoder forward pass (input: {X_tensor.shape}, output: {out_ae.shape})")
    
    # CNN+BiLSTM forward (reshaped with 60 features)
    X_seq = np.random.randn(10, 60).astype(np.float32).reshape(10, 30, 2)
    X_seq_tensor = torch.from_numpy(X_seq).to(X_tensor.device)
    cnn_lstm = ModelFactory.create_model('cnn_bilstm', input_dim=60, seq_length=30)
    cnn_lstm.eval()
    with torch.no_grad():
        out_cnn = cnn_lstm(X_seq_tensor)
    print(f"  ✓ CNN+BiLSTM forward pass (input: {X_seq_tensor.shape}, output: {out_cnn.shape})")
    
    # TCN forward
    tcn = ModelFactory.create_model('tcn', input_dim=60, seq_length=30)
    tcn.eval()
    with torch.no_grad():
        out_tcn = tcn(X_seq_tensor)
    print(f"  ✓ TCN forward pass (input: {X_seq_tensor.shape}, output: {out_tcn.shape})")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test training pipeline import
print("\n[8] Testing training pipeline import...")
try:
    from training import IDSTrainingPipeline
    print(f"  ✓ training.py imports successful")
except ImportError as e:
    print(f"  ✗ training.py import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All verification tests passed!")
print("=" * 70)
print("\nNext steps:")
print("  1. Install PyTorch: pip install -r requirements.txt")
print("  2. Run the GUI: python main.py")
print("  3. Configure and train a model")
