"""
Evaluation metrics and reporting for IDS.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

class IDSEvaluator:
    """Comprehensive IDS evaluation metrics."""

    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.results = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Compute evaluation metrics.
        
        y_true: true labels
        y_pred: predicted labels
        y_pred_proba: predicted probabilities (for ROC-AUC)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else [0, 0, 0, 0]

        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Specificity, Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall

        # ROC-AUC if probabilities provided
        roc_auc = None
        if y_pred_proba is not None:
            try:
                # Handle both binary and multi-class
                if y_pred_proba.ndim == 1:
                    roc_auc = roc_auc_score(y_true, y_pred_proba)
                else:
                    # Multi-class: use one-vs-rest
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', zero_division=0)
            except:
                roc_auc = None

        # False Positive Rate, False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        self.results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'fpr': float(fpr),
            'fnr': float(fnr),
            'confusion_matrix': cm.tolist(),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

        return self.results

    def get_formatted_report(self) -> str:
        """Get formatted evaluation report."""
        report = "╔════════════════════════════════════════════════════╗\n"
        report += "║         IDS EVALUATION REPORT                      ║\n"
        report += "╠════════════════════════════════════════════════════╣\n"
        report += f"║ Accuracy:        {self.results['accuracy']:.4f}                           ║\n"
        report += f"║ Precision:       {self.results['precision']:.4f}                           ║\n"
        report += f"║ Recall:          {self.results['recall']:.4f}                           ║\n"
        report += f"║ F1-Score:        {self.results['f1_score']:.4f}                           ║\n"
        report += f"║ Specificity:     {self.results['specificity']:.4f}                           ║\n"
        report += f"║ Sensitivity:     {self.results['sensitivity']:.4f}                           ║\n"
        if self.results['roc_auc'] is not None:
            report += f"║ ROC-AUC:         {self.results['roc_auc']:.4f}                           ║\n"
        report += f"║ FPR:             {self.results['fpr']:.4f}                           ║\n"
        report += f"║ FNR:             {self.results['fnr']:.4f}                           ║\n"
        report += "╠════════════════════════════════════════════════════╣\n"
        report += f"║ TP: {self.results['tp']:<6} TN: {self.results['tn']:<6} "
        report += f"FP: {self.results['fp']:<6} FN: {self.results['fn']:<6} ║\n"
        report += "╚════════════════════════════════════════════════════╝\n"
        return report

    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = np.array(self.results['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               ylabel='True label',
               xlabel='Predicted label')
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        return fig

    def plot_roc_curve(self, save_path: Optional[str] = None):
        """Plot ROC curve if probabilities available."""
        if self.y_pred_proba is None:
            logger.warning("No probabilities available for ROC curve")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            if self.y_pred_proba.ndim == 1:
                fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
                roc_auc = auc(fpr, tpr)
            else:
                # Multi-class: use one-vs-rest for class 1
                fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc="lower right")
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            return fig
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Performance Analysis
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceAnalyzer:
    """Analyze efficiency metrics."""

    def __init__(self):
        self.metrics = {}

    def analyze(self, model_size_mb: float, inference_time_ms: float,
               n_features_selected: int, total_features: int) -> Dict:
        """Analyze efficiency metrics."""
        self.metrics = {
            'model_size_mb': float(model_size_mb),
            'inference_time_ms': float(inference_time_ms),
            'n_features_selected': int(n_features_selected),
            'total_features': int(total_features),
            'feature_reduction_ratio': float(1 - n_features_selected / total_features) if total_features > 0 else 0,
            'is_lightweight': model_size_mb < 10 and inference_time_ms < 100
        }
        return self.metrics

    def get_formatted_report(self) -> str:
        """Get formatted efficiency report."""
        report = "╔════════════════════════════════════════════════════╗\n"
        report += "║         EFFICIENCY METRICS                         ║\n"
        report += "╠════════════════════════════════════════════════════╣\n"
        report += f"║ Model Size:              {self.metrics['model_size_mb']:.2f} MB         ║\n"
        report += f"║ Inference Time:          {self.metrics['inference_time_ms']:.2f} ms          ║\n"
        report += f"║ Features Selected:       {self.metrics['n_features_selected']}/{self.metrics['total_features']}              ║\n"
        report += f"║ Feature Reduction:       {self.metrics['feature_reduction_ratio']:.2%}                    ║\n"
        report += f"║ Is Lightweight:          {'YES' if self.metrics['is_lightweight'] else 'NO':<10}              ║\n"
        report += "╚════════════════════════════════════════════════════╝\n"
        return report
