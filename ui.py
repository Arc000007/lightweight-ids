"""
ui.py

PyQt5-based GUI for Lightweight IDS.
Professional UI with configuration panels and results dashboard.
"""

import sys
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QRadioButton, QButtonGroup, QCheckBox,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QLabel,
    QFileDialog, QProgressBar, QTextEdit, QMessageBox, QScrollArea,
    QTableWidget, QTableWidgetItem, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from training import IDSTrainingPipeline
from dataset_config import DATASET_PATHS, list_available_datasets

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Training Worker Thread
# ─────────────────────────────────────────────────────────────────────────────

class TrainingWorker(QThread):
    """Worker thread for model training."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, config: Dict, data_path: str):
        super().__init__()
        self.config = config
        self.data_path = data_path
        self.pipeline = None

    def run(self):
        """Run training pipeline."""
        try:
            self.emit_progress("Initializing pipeline...")
            self.config['data_path'] = self.data_path
            self.pipeline = IDSTrainingPipeline(self.config)

            self.emit_progress("Loading data...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.pipeline.load_data(max_rows=10000)

            self.emit_progress("Selecting features...")
            # Use actual feature names from preprocessor (not generic Feature_0, Feature_1...)
            feature_names = self.pipeline.preprocessor.feature_names
            X_train_fs, X_val_fs, importance = self.pipeline.select_features(
                X_train, y_train, X_val, y_val, feature_names
            )
            X_test_fs = self.pipeline.feature_selector.transform(X_test)

            self.emit_progress("Reducing dimensions...")
            X_train_red, X_val_red, X_test_red = self.pipeline.reduce_dimensions(
                X_train_fs, X_val_fs, X_test_fs
            )

            self.emit_progress("Augmenting data...")
            X_train_aug, y_train_aug = self.pipeline.augment_data(X_train_red, y_train)

            self.emit_progress("Training model...")
            self.pipeline.train_model(X_train_aug, y_train_aug, X_val_red, y_val)

            if self.config['model_type'] == 'attention_autoencoder':
                self.emit_progress("Setting up thresholding...")
                self.pipeline.setup_thresholding(X_train_aug, y_train_aug)

            self.emit_progress("Evaluating model...")
            results = self.pipeline.evaluate(X_test_red, y_test)

            self.finished.emit({
                'results': results,
                'summary': self.pipeline.get_summary(),
                'pipeline': self.pipeline
            })

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            self.error.emit(str(e))

    def emit_progress(self, message: str):
        """Emit progress message."""
        logger.info(message)
        self.progress.emit(message)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Panel
# ─────────────────────────────────────────────────────────────────────────────

class ConfigurationPanel(QWidget):
    """Configuration panel for all IDS settings."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()

        # Dataset Selection
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout()
        self.dataset_group = QButtonGroup()
        
        datasets = ["CICIoT2023", "TON_IoT", "IoT-23", "CICIDS2017", "CSE-CIC-IDS2018"]
        for i, dataset in enumerate(datasets):
            radio = QRadioButton(dataset)
            self.dataset_group.addButton(radio, i)
            dataset_layout.addWidget(radio)
            if i == 0:
                radio.setChecked(True)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # Data Augmentation
        augmentation_group = QGroupBox("Data Augmentation")
        augmentation_layout = QVBoxLayout()
        self.augmentation_checkbox = QCheckBox("Enable GAN-based Augmentation")
        augmentation_layout.addWidget(self.augmentation_checkbox)
        augmentation_group.setLayout(augmentation_layout)
        layout.addWidget(augmentation_group)

        # Feature Selection
        feature_group = QGroupBox("Feature Selection Method")
        feature_layout = QVBoxLayout()
        self.feature_group = QButtonGroup()
        
        features = ["Filter Methods", "Wrapper Methods", "Embedded Methods", "PSO", "ACO"]
        feature_values = ["filter", "wrapper", "embedded", "pso", "aco"]
        
        for i, (feature, value) in enumerate(zip(features, feature_values)):
            radio = QRadioButton(feature)
            radio.value = value
            self.feature_group.addButton(radio, i)
            feature_layout.addWidget(radio)
            if i == 0:
                radio.setChecked(True)
        
        feature_layout.addWidget(QLabel("Number of features:"))
        self.feature_k_spinbox = QSpinBox()
        self.feature_k_spinbox.setRange(5, 100)
        self.feature_k_spinbox.setValue(20)
        feature_layout.addWidget(self.feature_k_spinbox)
        
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)

        # Dimensionality Reduction
        reduction_group = QGroupBox("Dimensionality Reduction")
        reduction_layout = QVBoxLayout()
        self.reduction_group = QButtonGroup()
        
        reductions = ["None", "PCA", "Autoencoder"]
        reduction_values = ["none", "pca", "autoencoder"]
        
        for i, (reduction, value) in enumerate(zip(reductions, reduction_values)):
            radio = QRadioButton(reduction)
            radio.value = value
            self.reduction_group.addButton(radio, i)
            reduction_layout.addWidget(radio)
            if i == 0:
                radio.setChecked(True)
        
        reduction_layout.addWidget(QLabel("Reduction dimension:"))
        self.reduction_dim_spinbox = QSpinBox()
        self.reduction_dim_spinbox.setRange(5, 50)
        self.reduction_dim_spinbox.setValue(20)
        reduction_layout.addWidget(self.reduction_dim_spinbox)
        
        reduction_group.setLayout(reduction_layout)
        layout.addWidget(reduction_group)

        # Model Selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        self.model_group = QButtonGroup()
        
        models = ["Attention Autoencoder", "CNN + BiLSTM", "Temporal CNN (TCN)"]
        model_values = ["attention_autoencoder", "cnn_bilstm", "tcn"]
        
        for i, (model, value) in enumerate(zip(models, model_values)):
            radio = QRadioButton(model)
            radio.value = value
            self.model_group.addButton(radio, i)
            model_layout.addWidget(radio)
            if i == 0:
                radio.setChecked(True)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Hyperparameter Optimization
        hyperopt_group = QGroupBox("Hyperparameter Optimization")
        hyperopt_layout = QVBoxLayout()
        self.hyperopt_group = QButtonGroup()
        
        hyperopt_methods = ["Manual", "Grid Search", "Random Search", "PSO"]
        hyperopt_values = ["manual", "grid", "random", "pso"]
        
        for i, (method, value) in enumerate(zip(hyperopt_methods, hyperopt_values)):
            radio = QRadioButton(method)
            radio.value = value
            self.hyperopt_group.addButton(radio, i)
            hyperopt_layout.addWidget(radio)
            if i == 0:
                radio.setChecked(True)
        
        hyperopt_group.setLayout(hyperopt_layout)
        layout.addWidget(hyperopt_group)

        # Thresholding (for Autoencoder)
        threshold_group = QGroupBox("Thresholding (Autoencoder Only)")
        threshold_layout = QVBoxLayout()
        self.threshold_group = QButtonGroup()
        
        threshold_methods = ["Fixed", "Statistical (Mean + k*Std)", "PSO-optimized"]
        threshold_values = ["fixed", "statistical", "pso"]
        
        for i, (method, value) in enumerate(zip(threshold_methods, threshold_values)):
            radio = QRadioButton(method)
            radio.value = value
            self.threshold_group.addButton(radio, i)
            threshold_layout.addWidget(radio)
            if i == 1:
                radio.setChecked(True)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)

        # Training Parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(10, 500)
        self.epochs_spinbox.setValue(50)
        params_layout.addWidget(self.epochs_spinbox, 0, 1)
        
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(8, 256)
        self.batch_size_spinbox.setValue(32)
        params_layout.addWidget(self.batch_size_spinbox, 1, 1)
        
        params_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(1e-5, 1e-1)
        self.lr_spinbox.setValue(1e-3)  # Default: 0.001 (recommended)
        self.lr_spinbox.setSingleStep(1e-4)
        self.lr_spinbox.setDecimals(6)  # Show up to 6 decimal places
        self.lr_spinbox.setToolTip("Recommended: 1e-3 (0.001) for most models\nLower values = slower learning, higher values = faster learning")
        params_layout.addWidget(self.lr_spinbox, 2, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        layout.addStretch()
        self.setLayout(layout)

    def get_config(self) -> Dict:
        """Get current configuration."""
        selected_dataset = self.dataset_group.checkedButton()
        selected_feature = self.feature_group.checkedButton()
        selected_reduction = self.reduction_group.checkedButton()
        selected_model = self.model_group.checkedButton()
        selected_hyperopt = self.hyperopt_group.checkedButton()
        selected_threshold = self.threshold_group.checkedButton()

        return {
            'dataset_name': selected_dataset.text() if selected_dataset else 'CICIoT2023',
            'feature_selection_method': selected_feature.value if selected_feature else 'filter',
            'feature_k': self.feature_k_spinbox.value(),
            'augmentation_enabled': self.augmentation_checkbox.isChecked(),
            'dimensionality_reduction': selected_reduction.value if selected_reduction else 'none',
            'reduction_dim': self.reduction_dim_spinbox.value(),
            'model_type': selected_model.value if selected_model else 'attention_autoencoder',
            'hyperparameter_optimization': selected_hyperopt.value if selected_hyperopt else 'manual',
            'threshold_method': selected_threshold.value if selected_threshold else 'statistical',
            'epochs': self.epochs_spinbox.value(),
            'batch_size': self.batch_size_spinbox.value(),
            'learning_rate': self.lr_spinbox.value(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class ResultsDashboard(QWidget):
    """Dashboard for displaying training results."""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.results = None

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()

        # Tabs for different result views
        self.tabs = QTabWidget()

        # Evaluation Metrics Tab
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.tabs.addTab(self.metrics_text, "Evaluation Metrics")

        # Efficiency Metrics Tab
        self.efficiency_text = QTextEdit()
        self.efficiency_text.setReadOnly(True)
        self.tabs.addTab(self.efficiency_text, "Efficiency Metrics")

        # Feature Importance Tab
        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(2)
        self.importance_table.setHorizontalHeaderLabels(["Feature", "Importance Score"])
        self.tabs.addTab(self.importance_table, "Feature Importance")

        # Confusion Matrix Tab
        self.cm_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.tabs.addTab(self.cm_canvas, "Confusion Matrix")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def display_results(self, results: Dict, summary: Dict):
        """Display results."""
        self.results = results

        # Evaluation metrics
        metrics_text = "╔════════════════════════════════════════════════════╗\n"
        metrics_text += "║         IDS EVALUATION REPORT                      ║\n"
        metrics_text += "╠════════════════════════════════════════════════════╣\n"
        
        eval_data = results.get('evaluation', {})
        metrics_text += f"║ Accuracy:        {eval_data.get('accuracy', 0):.4f}                           ║\n"
        metrics_text += f"║ Precision:       {eval_data.get('precision', 0):.4f}                           ║\n"
        metrics_text += f"║ Recall:          {eval_data.get('recall', 0):.4f}                           ║\n"
        metrics_text += f"║ F1-Score:        {eval_data.get('f1_score', 0):.4f}                           ║\n"
        metrics_text += f"║ Specificity:     {eval_data.get('specificity', 0):.4f}                           ║\n"
        metrics_text += f"║ Sensitivity:     {eval_data.get('sensitivity', 0):.4f}                           ║\n"
        if eval_data.get('roc_auc'):
            metrics_text += f"║ ROC-AUC:         {eval_data.get('roc_auc'):.4f}                           ║\n"
        metrics_text += "╠════════════════════════════════════════════════════╣\n"
        metrics_text += f"║ TP: {eval_data.get('tp', 0):<6} TN: {eval_data.get('tn', 0):<6} "
        metrics_text += f"FP: {eval_data.get('fp', 0):<6} FN: {eval_data.get('fn', 0):<6} ║\n"
        metrics_text += "╚════════════════════════════════════════════════════╝\n"
        self.metrics_text.setText(metrics_text)

        # Efficiency metrics
        perf_data = results.get('performance', {})
        efficiency_text = "╔════════════════════════════════════════════════════╗\n"
        efficiency_text += "║         EFFICIENCY METRICS                         ║\n"
        efficiency_text += "╠════════════════════════════════════════════════════╣\n"
        efficiency_text += f"║ Model Size:              {perf_data.get('model_size_mb', 0):.2f} MB         ║\n"
        efficiency_text += f"║ Inference Time:          {perf_data.get('inference_time_ms', 0):.2f} ms          ║\n"
        efficiency_text += f"║ Features Selected:       {perf_data.get('n_features_selected', 0)}/{perf_data.get('total_features', 0)}              ║\n"
        efficiency_text += f"║ Feature Reduction:       {perf_data.get('feature_reduction_ratio', 0):.2%}                    ║\n"
        efficiency_text += f"║ Is Lightweight:          {'YES' if perf_data.get('is_lightweight') else 'NO':<10}              ║\n"
        efficiency_text += "╚════════════════════════════════════════════════════╝\n"
        self.efficiency_text.setText(efficiency_text)

        # Feature importance
        importance_scores = summary.get('feature_importance', {})
        self.importance_table.setRowCount(len(importance_scores))
        for i, (feature, score) in enumerate(sorted(importance_scores.items(), 
                                                     key=lambda x: x[1], reverse=True)[:20]):
            self.importance_table.setItem(i, 0, QTableWidgetItem(feature))
            self.importance_table.setItem(i, 1, QTableWidgetItem(f"{score:.4f}"))

        # Confusion matrix
        cm = np.array(eval_data.get('confusion_matrix', [[0, 0], [0, 0]]))
        ax = self.cm_canvas.figure.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               ylabel='True label',
               xlabel='Predicted label')
        
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_title('Confusion Matrix')
        self.cm_canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class IDSMainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.data_path = None
        self.training_worker = None
        self.init_ui()
        self.setStyleSheet(self.get_stylesheet())

    def init_ui(self):
        """Initialize UI."""
        self.setWindowTitle("Lightweight IDS - Configuration & Training")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel - Configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Dataset selector (primary method)
        dataset_quick_group = QGroupBox("Select Dataset")
        dataset_quick_layout = QVBoxLayout()
        dataset_quick_layout.addWidget(QLabel("Choose from available datasets:"))
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("-- Select a dataset --")
        
        available_datasets = list_available_datasets()
        for dataset_name, info in available_datasets.items():
            if info["exists"]:
                self.dataset_combo.addItem(f"✓ {dataset_name}", info["path"])
            else:
                self.dataset_combo.addItem(f"✗ {dataset_name} (not found)", info["path"])
        
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_selected)
        dataset_quick_layout.addWidget(self.dataset_combo)
        
        # Show selected path
        self.data_path_label = QLabel("No dataset selected")
        self.data_path_label.setWordWrap(True)
        self.data_path_label.setStyleSheet("color: #666; font-size: 10px;")
        dataset_quick_layout.addWidget(self.data_path_label)
        
        dataset_quick_group.setLayout(dataset_quick_layout)
        left_layout.addWidget(dataset_quick_group)

        # Configuration panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.config_panel = ConfigurationPanel()
        scroll.setWidget(self.config_panel)
        left_layout.addWidget(scroll)

        # Control buttons
        buttons_layout = QHBoxLayout()
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        buttons_layout.addWidget(self.train_button)

        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)

        left_layout.addLayout(buttons_layout)
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # Right panel - Results & Progress
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Hide until training starts
        progress_layout.addWidget(self.progress_bar)

        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(150)
        progress_layout.addWidget(self.progress_text)
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)

        # Results dashboard
        self.results_dashboard = ResultsDashboard()
        right_layout.addWidget(self.results_dashboard)

        right_panel.setLayout(right_layout)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def browse_data_path(self):
        """Browse for data path."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory", ""
        )
        if path:
            self.data_path = path
            self.data_path_label.setText(f"Path: {path}")
            self.dataset_combo.setCurrentIndex(0)  # Reset combo when manually browsing

    def on_dataset_selected(self, index):
        """Handle dataset selection from combo."""
        if index > 0:  # Skip the placeholder item
            dataset_path = self.dataset_combo.currentData()
            if dataset_path and os.path.isdir(dataset_path):
                self.data_path = dataset_path
                dataset_name = self.dataset_combo.currentText()
                self.data_path_label.setText(f"Path: {dataset_path}")
                logger.info(f"Selected dataset: {dataset_name}")

    def clear_data_path(self):
        """Clear the selected data path."""
        self.data_path = None
        self.data_path_label.setText("No path selected")
        self.dataset_combo.setCurrentIndex(0)

    def start_training(self):
        """Start model training."""
        if not self.data_path:
            QMessageBox.warning(self, "Error", "Please select a dataset path first")
            return

        config = self.config_panel.get_config()
        
        # Start training worker
        self.training_worker = TrainingWorker(config, self.data_path)
        self.training_worker.progress.connect(self.update_progress)
        self.training_worker.finished.connect(self.training_finished)
        self.training_worker.error.connect(self.training_error)

        self.progress_text.clear()
        self.progress_bar.setVisible(True)  # Show progress bar when starting
        self.progress_bar.setValue(0)
        self.train_button.setEnabled(False)
        self.training_worker.start()

    def update_progress(self, message: str):
        """Update progress display."""
        self.progress_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def training_finished(self, data: Dict):
        """Handle training completion."""
        results = data['results']
        summary = data['summary']
        self.pipeline = data['pipeline']

        self.results_dashboard.display_results(results, summary)
        self.progress_text.append("\n✓ Training completed successfully!")
        self.progress_bar.setValue(100)  # Set to 100%
        self.progress_bar.setVisible(False)  # Hide after completion
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(True)

        QMessageBox.information(self, "Success", "Model training completed successfully!")

    def training_error(self, error: str):
        """Handle training error."""
        self.progress_text.append(f"\n✗ Error: {error}")
        self.progress_bar.setVisible(False)  # Hide on error
        self.train_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Training failed: {error}")

    def save_results(self):
        """Save results to JSON."""
        if not hasattr(self, 'pipeline'):
            QMessageBox.warning(self, "Error", "No results to save")
            return

        path = QFileDialog.getSaveFileName(self, "Save Results", "", "JSON Files (*.json)")[0]
        if path:
            summary = self.pipeline.get_summary()
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            QMessageBox.information(self, "Success", f"Results saved to {path}")

    def get_stylesheet(self) -> str:
        """Get application stylesheet."""
        return """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            color: #333;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 0.5ex;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #1565C0;
        }
        QRadioButton, QCheckBox {
            spacing: 5px;
        }
        QTextEdit {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        """


# ─────────────────────────────────────────────────────────────────────────────
# Application Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = IDSMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
