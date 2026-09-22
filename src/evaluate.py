"""
Model evaluation for the Support Ticket Classification pipeline.

Generates comprehensive evaluation metrics, confusion matrices,
and class-wise performance breakdowns for both classifiers.
All artifacts are saved to the reports/ directory.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# Consistent plot styling
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def evaluate_model(
    model,
    X_test,
    y_test: np.ndarray,
    label_names: List[str],
    task_name: str,
) -> Dict:
    """
    Evaluate a trained classifier on the test set and save artifacts.

    Computes:
    - Overall metrics: accuracy, precision, recall, F1 (macro and weighted)
    - Confusion matrix (saved as PNG heatmap)
    - Per-class precision/recall/F1/support table (saved as CSV and PNG)

    Args:
        model: Trained classifier with predict() method.
        X_test: Test feature matrix.
        y_test: True labels for test set.
        label_names: Ordered list of class label names.
        task_name: Human-readable name ('Category' or 'Priority').

    Returns:
        Dictionary of evaluation metrics for storage in metadata.
    """
    logger.info("Evaluating %s classifier on test set (%d samples)...", task_name, X_test.shape[0])

    y_pred = model.predict(X_test)

    # --- Overall metrics ---
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    logger.info(
        "  %s Results — Accuracy: %.4f | F1 (macro): %.4f | F1 (weighted): %.4f",
        task_name, metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"],
    )

    # --- Classification report (per-class) ---
    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T
    report_csv_path = REPORTS_DIR / f"{task_name.lower()}_classification_report.csv"
    report_df.to_csv(report_csv_path)
    logger.info("  Saved classification report to %s", report_csv_path)

    # Print report to console
    report_text = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0
    )
    logger.info("\n%s Classification Report:\n%s", task_name, report_text)

    # --- Confusion matrix heatmap ---
    _plot_confusion_matrix(y_test, y_pred, label_names, task_name)

    # --- Per-class performance bar chart ---
    _plot_class_performance(report_dict, label_names, task_name)

    # Store per-class metrics in the result dict
    metrics["per_class"] = {}
    for label in label_names:
        if label in report_dict:
            metrics["per_class"][label] = {
                "precision": float(report_dict[label]["precision"]),
                "recall": float(report_dict[label]["recall"]),
                "f1": float(report_dict[label]["f1-score"]),
                "support": int(report_dict[label]["support"]),
            }

    return metrics


def _plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    task_name: str,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=label_names)

    # Normalize for better readability (show percentages)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(label_names) * 1.8), max(6, len(label_names) * 0.8)))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0], cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title(f"{task_name} — Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized (percentages)
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1], cbar_kws={"shrink": 0.8},
        vmin=0, vmax=1,
    )
    axes[1].set_title(f"{task_name} — Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()

    path = REPORTS_DIR / f"{task_name.lower()}_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved confusion matrix to %s", path)


def _plot_class_performance(
    report_dict: Dict,
    label_names: List[str],
    task_name: str,
) -> None:
    """Generate and save a per-class performance bar chart."""
    class_data = []
    for label in label_names:
        if label in report_dict:
            class_data.append({
                "Class": label,
                "Precision": report_dict[label]["precision"],
                "Recall": report_dict[label]["recall"],
                "F1-Score": report_dict[label]["f1-score"],
            })

    if not class_data:
        return

    df = pd.DataFrame(class_data)
    df_melted = df.melt(id_vars="Class", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(max(10, len(label_names) * 1.5), 6))

    bar_width = 0.25
    x = np.arange(len(label_names))

    for i, metric in enumerate(["Precision", "Recall", "F1-Score"]):
        values = df[metric].values
        bars = ax.bar(x + i * bar_width, values, bar_width, label=metric, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title(f"{task_name} — Per-Class Performance")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    path = REPORTS_DIR / f"{task_name.lower()}_class_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved class performance chart to %s", path)


def generate_summary_report(
    category_metrics: Dict,
    priority_metrics: Dict,
    category_cv: Dict,
    priority_cv: Dict,
) -> None:
    """
    Generate a combined markdown evaluation report.

    Args:
        category_metrics: Test-set evaluation metrics for category classifier.
        priority_metrics: Test-set evaluation metrics for priority classifier.
        category_cv: Cross-validation results for category models.
        priority_cv: Cross-validation results for priority models.
    """
    report_lines = [
        "# Model Evaluation Report",
        f"",
        "## Category Classifier",
        "",
        "### Cross-Validation Results",
        "",
        "| Model | F1 (macro) | Accuracy |",
        "|-------|-----------|----------|",
    ]

    for name, scores in category_cv.items():
        if "error" in scores:
            report_lines.append(f"| {name} | ERROR | ERROR |")
        else:
            report_lines.append(
                f"| {name} | {scores['mean_f1']:.4f} ± {scores['std_f1']:.4f} "
                f"| {scores['mean_accuracy']:.4f} ± {scores['std_accuracy']:.4f} |"
            )

    report_lines.extend([
        "",
        "### Test Set Results",
        "",
        f"- **Accuracy**: {category_metrics['accuracy']:.4f}",
        f"- **F1 (macro)**: {category_metrics['f1_macro']:.4f}",
        f"- **F1 (weighted)**: {category_metrics['f1_weighted']:.4f}",
        f"- **Precision (macro)**: {category_metrics['precision_macro']:.4f}",
        f"- **Recall (macro)**: {category_metrics['recall_macro']:.4f}",
        "",
        "---",
        "",
        "## Priority Classifier",
        "",
        "### Cross-Validation Results",
        "",
        "| Model | F1 (macro) | Accuracy |",
        "|-------|-----------|----------|",
    ])

    for name, scores in priority_cv.items():
        if "error" in scores:
            report_lines.append(f"| {name} | ERROR | ERROR |")
        else:
            report_lines.append(
                f"| {name} | {scores['mean_f1']:.4f} ± {scores['std_f1']:.4f} "
                f"| {scores['mean_accuracy']:.4f} ± {scores['std_accuracy']:.4f} |"
            )

    report_lines.extend([
        "",
        "### Test Set Results",
        "",
        f"- **Accuracy**: {priority_metrics['accuracy']:.4f}",
        f"- **F1 (macro)**: {priority_metrics['f1_macro']:.4f}",
        f"- **F1 (weighted)**: {priority_metrics['f1_weighted']:.4f}",
        f"- **Precision (macro)**: {priority_metrics['precision_macro']:.4f}",
        f"- **Recall (macro)**: {priority_metrics['recall_macro']:.4f}",
    ])

    report_path = REPORTS_DIR / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info("Saved evaluation report to %s", report_path)
