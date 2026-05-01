"""
models — classifier definitions and evaluation helpers for GTZAN genre
classification.

Shared utilities (plotting, result logging) live here; individual model
definitions live in their own submodules (gnb, knn, svm, rf, xgb).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from config import PLOTS_MODELS
from utils import get_logger

log = get_logger("models")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130

_plots_dir = PLOTS_MODELS


def set_plots_dir(path):
    """Override the directory used by :func:`savefig`."""
    global _plots_dir
    _plots_dir = path
    os.makedirs(path, exist_ok=True)


def savefig(name):
    """Save the current figure to the models plot directory and close it."""
    path = os.path.join(_plots_dir, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path}")


def confusion_matrix_plot(y_true, y_pred, classes, title, fname, cmap="Blues"):
    """Generate and save a confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=classes,
        cmap=cmap, ax=ax, colorbar=True,
    )
    ax.set_title(title, fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    savefig(fname)
