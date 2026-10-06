"""
Quick Start Guide for Lightweight IDS System

This file provides step-by-step instructions to get up and running quickly.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: INSTALLATION
# ─────────────────────────────────────────────────────────────────────────────

"""
1. Open terminal/PowerShell in the project directory:
   cd path/to/project

2. Create virtual environment (optional but recommended):
   python -m venv .venv
   .venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Verify installation:
   python test_setup.py
   
   You should see all tests pass.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PREPARING YOUR DATASET
# ─────────────────────────────────────────────────────────────────────────────

"""
Supported dataset formats:
- CSV files (.csv)
- Parquet files (.parquet)
- Directory with multiple CSV/Parquet files

Dataset format requirements:
- Must have a label column (dataset-specific names)
- Numeric features
- No identifier columns (IP addresses, timestamps, etc.) - automatically filtered

Supported datasets built-in:
1. CICIoT2023     → label_col: 'label', benign_label: 'BenignTraffic'
2. TON_IoT        → label_col: 'type', benign_label: 'normal'
3. IoT-23         → label_col: 'label', benign_label: 'Benign'
4. CICIDS2017     → label_col: ' Label', benign_label: 'BENIGN'
5. CSE-CIC-IDS2018→ label_col: 'Label', benign_label: 'Benign'

Example: If using CICIoT2023 dataset from:
  https://www.unb.ca/cic/datasets/ciciot2023.html
  
Place the CSV file in: path/to/project\data\
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: LAUNCHING THE GUI
# ─────────────────────────────────────────────────────────────────────────────

"""
Run the main application:
   python main.py

This opens the PyQt5 GUI with:
- Left panel: Configuration options
- Right panel: Results and training progress
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: CONFIGURING FOR YOUR USE CASE
# ─────────────────────────────────────────────────────────────────────────────

"""
FOR LIGHTWEIGHT IoT DEPLOYMENT:
- Feature Selection: Filter Methods (fast)
- Features to select: 10-15
- Data Augmentation: OFF
- Dimensionality Reduction: PCA
- Model: TCN (most efficient)
- Hyperparameter Optimization: Manual
- Epochs: 30-50
- Batch Size: 64

FOR HIGH ACCURACY RESEARCH:
- Feature Selection: PSO (slow but thorough)
- Features to select: 40-50
- Data Augmentation: ON (GAN)
- Dimensionality Reduction: Autoencoder
- Model: CNN + BiLSTM (highest accuracy)
- Hyperparameter Optimization: PSO
- Epochs: 100+
- Batch Size: 32

FOR QUICK TESTING/PROTOTYPING:
- Feature Selection: Filter Methods
- Features to select: 10
- Data Augmentation: OFF
- Dimensionality Reduction: None
- Model: TCN
- Hyperparameter Optimization: Manual
- Epochs: 5-10 (for testing)
- Batch Size: 128
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: UNDERSTANDING THE WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

"""
The system follows this pipeline:

1. DATASET SELECTION
   → Choose from 5 supported datasets

2. PREPROCESSING
   → Load, clean, normalize, split (train/val/test)
   → Automatically handles:
     - Missing values
     - Duplicate rows
     - Constant features
     - Feature scaling

3. DATA AUGMENTATION (Optional)
   → Uses WGAN-GP to generate synthetic samples
   → Helps with class imbalance
   → Only if "Enable GAN" is checked

4. FEATURE SELECTION
   Five methods available:
   a) Filter Methods
      - Fast, stateless
      - Uses MI + ANOVA
      - Selects top-k features
      - ✓ Good for quick prototyping
      
   b) Wrapper Methods
      - Slower, uses model feedback
      - Recursive Feature Elimination
      - Considers feature interactions
      
   c) Embedded Methods
      - Uses Random Forest importance
      - Selects based on model training
      
   d) PSO (Particle Swarm Optimization)
      - Metaheuristic search
      - Good balance of speed/quality
      - 🔥 Recommended for research
      
   e) ACO (Ant Colony Optimization)
      - Biologically-inspired
      - Pheromone-guided search
      - Similar quality to PSO

5. DIMENSIONALITY REDUCTION (Optional)
   - None: Keep all features
   - PCA: Classic linear reduction
   - Autoencoder: Learned nonlinear compression

6. MODEL SELECTION
   Three architectures:
   
   a) Attention Autoencoder
      - Unsupervised/semi-supervised
      - Good for anomaly detection
      - Uses reconstruction error threshold
      
   b) CNN + BiLSTM
      - Highest accuracy
      - Combines spatial + temporal
      - Higher computational cost
      
   c) TCN (Temporal CNN)
      - Lightweight & efficient
      - Dilated convolutions
      - ✓ Recommended for IoT deployment

7. HYPERPARAMETER OPTIMIZATION
   - Manual: Use configured parameters
   - Grid Search: Exhaustive search
   - Random Search: Stochastic sampling
   - PSO: Optimize via swarm intelligence

8. THRESHOLDING (Autoencoder only)
   For autoencoder-based IDS:
   - Fixed: User-defined threshold
   - Statistical: Mean + k·std
   - PSO-optimized: Maximize F1-score

9. TRAINING
   Model is trained on augmented data

10. EVALUATION
    Metrics computed on test set:
    - Classification: Accuracy, Precision, Recall, F1
    - Security: TPR, FPR, ROC-AUC
    - Efficiency: Model size, inference time
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: INTERPRETING RESULTS
# ─────────────────────────────────────────────────────────────────────────────

"""
Results tabs in GUI:

1. EVALUATION METRICS
   - Accuracy: Overall correctness
   - Precision: False alarm rate (lower is better)
   - Recall: Detection rate (higher is better)
   - F1-Score: Harmonic mean (balance)
   - Specificity: True negative rate
   - ROC-AUC: Area under ROC curve

2. EFFICIENCY METRICS
   - Model Size: Total parameters in MB
   - Inference Time: Average latency per sample
   - Features Selected: Dimensionality reduction
   - Lightweight: Is lightweight for deployment? (size < 10MB, latency < 100ms)

3. FEATURE IMPORTANCE
   - Top 20 features with scores
   - Higher score = more important
   - Use for domain interpretation

4. CONFUSION MATRIX
   - TP (True Positives): Correctly detected attacks
   - TN (True Negatives): Correctly identified benign
   - FP (False Positives): False alarms
   - FN (False Negatives): Missed attacks
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: ADVANCED USAGE
# ─────────────────────────────────────────────────────────────────────────────

"""
Using configuration presets:

from config_templates import get_preset

# Load preset
config = get_preset('lightweight')  # or 'high_accuracy', 'balanced', 'quick_test'

# Modify as needed
config['feature_k'] = 20
config['epochs'] = 50

# Use in training
from training import IDSTrainingPipeline
config['data_path'] = 'path/to/dataset.csv'
pipeline = IDSTrainingPipeline(config)
# ... rest of training code
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: SAVING & EXPORTING RESULTS
# ─────────────────────────────────────────────────────────────────────────────

"""
From GUI:
1. After training completes, click "Save Results"
2. Choose location and filename
3. Results saved as JSON with:
   - Configuration
   - Evaluation metrics
   - Performance metrics
   - Feature importance scores

From code:
pipeline.save_model('models/my_ids_model.h5')

import json
summary = pipeline.get_summary()
with open('results.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: COMMON ISSUES & SOLUTIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
Problem: "ModuleNotFoundError: No module named 'tensorflow'"
Solution: pip install tensorflow

Problem: "Out of Memory" during training
Solution:
  - Reduce batch_size (32 → 64)
  - Reduce feature_k (50 → 20)
  - Use simpler model (TCN instead of CNN+BiLSTM)
  - Use GPU acceleration (CUDA)

Problem: "Feature selection is very slow"
Solution:
  - Use Filter methods instead of PSO/ACO
  - Reduce n_particles/n_ants
  - Reduce n_iter (iterations)
  - Reduce feature_k

Problem: "Poor model accuracy"
Solution:
  - Enable data augmentation
  - Increase feature_k
  - Try different feature selection method
  - Increase epochs
  - Use hyperparameter optimization

Problem: "No GPU detected"
Solution:
  - Ensure CUDA 11.x is installed
  - Verify TensorFlow GPU: python -c "import tensorflow; print(tensorflow.test.is_built_with_cuda())"
"""

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT FILE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

"""
DeepLearning/
│
├── [MAIN ENTRY POINT]
├── main.py                   ← Run this to start the GUI
│
├── [CONFIGURATION]
├── config_templates.py       ← Predefined configurations
├── requirements.txt          ← Python dependencies
├── README.md                 ← Full documentation
├── QUICKSTART.md             ← This file
│
├── [CORE MODULES]
├── ui.py                     ← PyQt5 graphical interface
├── training.py               ← Training pipeline orchestrator
│
├── [DATA PROCESSING]
├── Preprocessing.py          ← Dataset loading & cleaning
├── Feature_selection.py      ← 5 feature selection methods
├── Augmentation.py           ← WGAN-GP data augmentation
│
├── [MODELS & TECHNIQUES]
├── models.py                 ← 3 deep learning architectures
├── dimensionality_reduction.py → PCA & Autoencoder
├── hyperparameter_optimization.py → Grid/Random/PSO search
├── thresholding.py           ← Threshold computation
├── evaluation.py             ← Evaluation metrics
│
├── [TESTING]
└── test_setup.py             ← Verify installation

Output files (created after running):
├── ids_training.log          ← Training logs
├── models/                   ← Saved model files
└── results/                  ← JSON results
"""

print(__doc__)
