"""
Example configuration templates for quick setup.
Copy and modify these configurations for your use case.
"""

import json

# ─────────────────────────────────────────────────────────────────────────────
# Configuration Presets
# ─────────────────────────────────────────────────────────────────────────────

LIGHTWEIGHT_CONFIG = {
    'dataset_name': 'Auto-detect',
    'feature_selection_method': 'filter',
    'feature_k': 15,
    'augmentation_enabled': False,
    'dimensionality_reduction': 'pca',
    'reduction_dim': 15,
    'model_type': 'tcn',
    'hyperparameter_optimization': 'manual',
    'threshold_method': 'statistical',
    'epochs': 30,
    'batch_size': 64,
    'learning_rate': 1e-3,
}

HIGH_ACCURACY_CONFIG = {
    'dataset_name': 'Auto-detect',
    'feature_selection_method': 'pso',
    'feature_k': 40,
    'augmentation_enabled': True,
    'dimensionality_reduction': 'autoencoder',
    'reduction_dim': 30,
    'model_type': 'cnn_bilstm',
    'hyperparameter_optimization': 'pso',
    'threshold_method': 'pso',
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 5e-4,
}

BALANCED_CONFIG = {
    'dataset_name': 'Auto-detect',
    'feature_selection_method': 'embedded',
    'feature_k': 25,
    'augmentation_enabled': True,
    'dimensionality_reduction': 'pca',
    'reduction_dim': 20,
    'model_type': 'attention_autoencoder',
    'hyperparameter_optimization': 'grid',
    'threshold_method': 'statistical',
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-3,
}

QUICK_TEST_CONFIG = {
    'dataset_name': 'Auto-detect',
    'feature_selection_method': 'filter',
    'feature_k': 10,
    'augmentation_enabled': False,
    'dimensionality_reduction': 'none',
    'reduction_dim': 10,
    'model_type': 'tcn',
    'hyperparameter_optimization': 'manual',
    'threshold_method': 'fixed',
    'epochs': 5,
    'batch_size': 128,
    'learning_rate': 1e-3,
}

# ─────────────────────────────────────────────────────────────────────────────
# Save/Load Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_config(config: dict, filepath: str):
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {filepath}")

def load_config(filepath: str) -> dict:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Config loaded from {filepath}")
    return config

def get_preset(preset_name: str) -> dict:
    """Get predefined configuration preset."""
    presets = {
        'lightweight': LIGHTWEIGHT_CONFIG,
        'high_accuracy': HIGH_ACCURACY_CONFIG,
        'balanced': BALANCED_CONFIG,
        'quick_test': QUICK_TEST_CONFIG,
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    return presets[preset_name].copy()

if __name__ == '__main__':
    # Example: Save all presets
    presets = {
        'lightweight': LIGHTWEIGHT_CONFIG,
        'high_accuracy': HIGH_ACCURACY_CONFIG,
        'balanced': BALANCED_CONFIG,
        'quick_test': QUICK_TEST_CONFIG,
    }
    
    for name, config in presets.items():
        save_config(config, f'configs/{name}_config.json')
