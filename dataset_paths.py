"""
Dataset paths and configurations for local datasets.
Maps dataset names to their actual directory locations and loading strategies.
"""

from pathlib import Path

# Base path for all datasets
BASE_DATA_PATH = Path(__file__).resolve().parent

# Dataset configurations with actual directory paths
DATASET_PATHS = {
    "CIC-IDS-2017": {
        "path": BASE_DATA_PATH / "CIC-IDS- 2017",
        "type": "directory",  # Load all CSVs in directory
        "label_col": " Label",
        "benign_label": "BENIGN",
        "drop_cols": ["Unnamed: 0"],
        "description": "Canadian Institute for Cybersecurity IDS 2017"
    },
    "CSE-CIC-IDS2018": {
        "path": BASE_DATA_PATH / "CSE-CIC-IDS2018",
        "type": "directory",  # Load all CSVs in directory
        "label_col": "Label",
        "benign_label": "Benign",
        "drop_cols": ["Timestamp"],
        "description": "Communications Security Establishment IDS 2018"
    },
    "CICIoT2023": {
        "path": BASE_DATA_PATH / "CICIOT2023",
        "type": "nested_directory",  # Has train/test/validation subdirs
        "label_col": "label",
        "benign_label": "BenignTraffic",
        "drop_cols": ["flow_id", "src_ip", "dst_ip", "src_mac", "dst_mac", "timestamp"],
        "subdirs": ["train", "test", "validation"],
        "description": "Canadian Institute for Cybersecurity IoT 2023"
    },
    "Edge-IIoTset": {
        "path": BASE_DATA_PATH / "Edge-IIoTset" / "Edge-IIoTset dataset",
        "type": "attack_normal",  # Has Attack traffic and Normal traffic subdirs
        "label_col": "Attack_type",
        "benign_label": "Normal",
        "drop_cols": [],
        "attack_dir": "Attack traffic",
        "normal_dir": "Normal traffic",
        "description": "Edge Industrial IoT Security Dataset"
    },
    "TON_IoT": {
        "path": BASE_DATA_PATH / "TON_IOT",
        "type": "nested_directory",
        "label_col": "type",
        "benign_label": "normal",
        "drop_cols": ["ts", "uid"],
        "description": "TON_IoT Dataset"
    }
}


def get_available_datasets():
    """Get list of datasets that actually exist in the filesystem."""
    available = {}
    for name, config in DATASET_PATHS.items():
        if config["path"].exists():
            available[name] = config
    return available


def get_dataset_config(dataset_name: str):
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_PATHS[dataset_name].copy()
    if not config["path"].exists():
        raise FileNotFoundError(f"Dataset path not found: {config['path']}")
    
    return config


def print_available_datasets():
    """Print list of available datasets."""
    available = get_available_datasets()
    print("\n" + "="*60)
    print("AVAILABLE DATASETS")
    print("="*60)
    for name, config in available.items():
        print(f"\n✓ {name}")
        print(f"  Path: {config['path']}")
        print(f"  Type: {config['type']}")
        print(f"  Description: {config.get('description', 'N/A')}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print_available_datasets()
