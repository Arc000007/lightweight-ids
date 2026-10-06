"""
Dataset paths and configuration for available IDS datasets.
Update these paths based on your local dataset locations.
"""

from pathlib import Path

# Base dataset directory
BASE_DATASET_DIR = Path(__file__).parent / "data"

# Dataset paths - update these to match your system
DATASET_PATHS = {
    "CICIoT2023": {
        "path": Path(__file__).parent / "CICIOT2023",
        "description": "CICIoT2023 - Canadian Institute for Cybersecurity IoT Dataset 2023",
        "file_pattern": "*.csv",
        "has_subdirs": True,
    },
    "TON_IoT": {
        "path": Path(__file__).parent / "TON_IOT",
        "description": "TON_IoT - Toniot IoT Security Dataset",
        "file_pattern": "*.csv",
        "has_subdirs": True,
    },
    "CICIDS2017": {
        "path": Path(__file__).parent / "CIC-IDS- 2017",
        "description": "CICIDS2017 - Canadian Institute for Cybersecurity IDS 2017",
        "file_pattern": "*.csv",
        "has_subdirs": False,
    },
    "CSE-CIC-IDS2018": {
        "path": Path(__file__).parent / "CSE-CIC-IDS2018",
        "description": "CSE-CIC-IDS2018 - Communications Security Establishment IDS 2018",
        "file_pattern": "*.csv",
        "has_subdirs": False,
    },
    "Edge-IIoTset": {
        "path": Path(__file__).parent / "Edge-IIoTset",
        "description": "Edge-IIoTset - Edge Industrial IoT Security Dataset",
        "file_pattern": "*.csv",
        "has_subdirs": True,
    },
}

def get_dataset_path(dataset_name: str):
    """Get the path for a dataset."""
    if dataset_name in DATASET_PATHS:
        return DATASET_PATHS[dataset_name]["path"]
    return None

def list_available_datasets():
    """List all available datasets and their status."""
    available = {}
    for name, config in DATASET_PATHS.items():
        path = config["path"]
        exists = path.exists()
        available[name] = {
            "path": str(path),
            "exists": exists,
            "description": config["description"]
        }
    return available

if __name__ == "__main__":
    import json
    datasets = list_available_datasets()
    print(json.dumps(datasets, indent=2))
