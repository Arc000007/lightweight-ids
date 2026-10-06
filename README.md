# Lightweight Intrusion Detection System

A desktop application for experimenting with machine learning pipelines for network intrusion detection. It combines a PyQt5 interface with a PyTorch training pipeline, feature selection, optional dimensionality reduction and augmentation, and evaluation metrics.

> **Status:** Research and learning project. Results depend on the dataset, preprocessing, configuration, and evaluation setup. Treat reported scores as experimental results, not evidence of production security effectiveness.

## At a glance

- Configure and launch model training from a desktop GUI.
- Compare filter, wrapper, embedded, particle swarm (PSO), and ant colony (ACO) feature selection.
- Choose PCA, autoencoder-based reduction, or no dimensionality reduction.
- Experiment with an attention autoencoder, CNN–BiLSTM, or temporal convolutional network (TCN).
- Optionally use WGAN-GP augmentation for imbalanced training data.
- Review classification and efficiency metrics, confusion matrix, training progress, and selected feature importance.
- Use CUDA when a compatible PyTorch installation and GPU are available; CPU execution is also supported.

## Pipeline

```text
CSV dataset
    ↓
Load, clean, encode, scale, and split
    ↓
Optional WGAN-GP augmentation
    ↓
Feature selection → optional dimensionality reduction
    ↓
Train selected PyTorch model
    ↓
Evaluate and inspect results in the GUI
```

## Models and methods

| Area | Options in the project |
| --- | --- |
| Feature selection | Filter, wrapper, embedded, PSO, ACO |
| Dimensionality reduction | None, PCA, autoencoder |
| Models | Attention autoencoder, CNN–BiLSTM, TCN |
| Augmentation | Optional WGAN-GP |
| Thresholding | Fixed, statistical, or PSO-based options for autoencoder workflows |
| Evaluation | Accuracy, precision, recall, F1, confusion matrix, and performance measurements |

The application is designed for binary benign-versus-attack classification. Confirm the dataset label handling and model behavior for your chosen data before interpreting results.

## Requirements

- Python 3.9 or later is recommended.
- Dependencies listed in [`requirements.txt`](requirements.txt).
- A compatible PyTorch/CUDA setup is optional; the project can use CPU when CUDA is unavailable.
- Network intrusion datasets are not included. Download them from their respective maintainers and follow their licenses and terms of use.

## Installation

```bash
git clone <your-repository-url>
cd <repository-directory>
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and launch the application:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

The application entry point is `main.py`. It initializes PyTorch, logs device information, and starts the PyQt5 interface.

## Dataset setup

Dataset files are intentionally kept outside source control. The repository does not bundle CIC, TON_IoT, Edge-IIoTset, or other dataset files.

1. Obtain a supported dataset from its official source and check its usage terms.
2. Put the extracted CSV files in the expected local directory, or update the paths in `dataset_config.py` for your machine.
3. Check that the chosen dataset has the label column and label values expected by `Preprocessing.py`.
4. Launch `python main.py` and select/configure a dataset in the application.

The repository contains both `dataset_config.py` and `dataset_paths.py`; dataset path and label settings are not fully centralized. If a dataset is not detected or a label cannot be found, inspect both configuration and preprocessing logic and align them with your local dataset layout. Dataset names shown in the GUI should also be checked against the configured directories before training.

## Using the application

1. Start the program with `python main.py`.
2. Choose a dataset and configure the feature selection method and feature count.
3. Optionally enable augmentation and dimensionality reduction.
4. Choose a model and training settings.
5. Start training and follow progress in the results panel.
6. Review evaluation metrics, confusion matrix, feature importance, and efficiency measurements.
7. Save results from the GUI when you want to keep an experiment summary.

For a quick initial run, use a small dataset sample, filter-based feature selection, no augmentation, and a small number of epochs. Increase data volume and training cost after verifying that the dataset loads and labels are interpreted correctly.

## Project layout

```text
.
├── main.py                         # Application entry point
├── ui.py                           # PyQt5 configuration and results interface
├── training.py                     # End-to-end training pipeline
├── Preprocessing.py                # Dataset loading and preprocessing
├── Feature_selection.py            # Feature selection algorithms
├── Augmentation_pytorch.py         # WGAN-GP augmentation
├── dimensionality_reduction_pytorch.py # PCA and autoencoder reduction
├── models.py                       # Model factory used by the training pipeline
├── models_pytorch.py               # PyTorch model implementations
├── thresholding.py                 # Anomaly threshold strategies
├── evaluation.py                   # Classification and performance metrics
├── dataset_config.py               # Dataset locations and UI dataset metadata
├── dataset_paths.py                # Dataset path/label configuration
├── config_templates.py              # Configuration presets
├── requirements.txt                # Python dependencies
└── test_setup.py                   # Environment/setup checks
```

Additional migration and analysis notes are available in `PYTORCH_MIGRATION.md`, `DETAILED_ANALYSIS.md`, and related project notes.

## Troubleshooting

**A dataset is missing from the application or cannot be loaded**  
Check the dataset directory configured in `dataset_config.py`, then check the matching dataset label and path in `dataset_paths.py` and `Preprocessing.py`. Paths and labels must match the files you downloaded.

**Training runs out of memory or is too slow**  
Try fewer input rows, a smaller feature count, a smaller batch size, or a less expensive feature-selection method. PSO/ACO and GAN augmentation can add substantial training time.

**CUDA is unavailable**  
The program can use CPU. For GPU use, install a PyTorch build compatible with the host's CUDA driver and verify it with `python -c "import torch; print(torch.cuda.is_available())"`.

**Qt or plotting dependencies fail to import**  
Confirm that the active Python environment is the one where `requirements.txt` was installed. On headless systems, a desktop display server may also be required for the GUI.

## Responsible evaluation

Intrusion detection datasets can be imbalanced, duplicated, or collected under conditions that differ from deployment. Report per-class metrics and confusion matrices, use a held-out test set, and avoid tuning on test labels. Benchmark latency and model size on the target hardware before making deployment claims. This project is an experimental tool and is not a substitute for a complete security monitoring system.

## License

No license file is currently included. Add a `LICENSE` file before redistributing the project so others know what permissions apply.

## Contributing

Issues and pull requests are welcome. Include the Python version, operating system, dataset/configuration details (without sharing restricted data), and relevant error output when reporting a problem.
