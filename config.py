"""
config.py — shared constants and paths for the GTZAN genre classification project.
Edit DATA_ROOT to point to your local dataset copy.
"""

import os
import multiprocessing
from datetime import datetime

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.join(os.path.dirname(__file__), os.pardir, "dataset", "Data")
GENRES_DIR = os.path.join(DATA_ROOT, "genres_original")
CORRUPT    = {"jazz.00054.wav"}

# ── Audio ─────────────────────────────────────────────────────────────────────
SR       = 22050
DURATION = 30
N_MFCC   = 20

# ── Outputs ───────────────────────────────────────────────────────────────────
OUT_ROOT        = os.path.join(os.path.dirname(__file__), "outputs")
PLOTS_EDA       = os.path.join(OUT_ROOT, "plots", "eda")
CACHE_DIR       = os.path.join(OUT_ROOT, "cache")
RUNS_ROOT       = os.path.join(OUT_ROOT, "runs")

FEAT_CACHE      = os.path.join(CACHE_DIR, "features.csv")
AUG_CACHE       = os.path.join(CACHE_DIR, "augmented_features.csv")

# Default (non-run-versioned) paths — used as fallbacks and by EDA / features
PLOTS_MODELS    = os.path.join(OUT_ROOT, "plots", "models")
RESULTS_DIR     = os.path.join(OUT_ROOT, "results")
MODELS_DIR      = os.path.join(OUT_ROOT, "models")
SPLITS_DIR      = os.path.join(OUT_ROOT, "splits")
CV_RESULTS_CSV  = os.path.join(RESULTS_DIR, "cv_results.csv")
RESULTS_CSV     = os.path.join(RESULTS_DIR, "final_results.csv")
TRAIN_SPLIT     = os.path.join(SPLITS_DIR, "train.csv")
TEST_SPLIT      = os.path.join(SPLITS_DIR, "test.csv")
LOG_FILE        = os.path.join(OUT_ROOT, "run.log")

# ── Feature extraction ────────────────────────────────────────────────────────
AUG_TYPES       = ["time_stretch", "pitch_shift", "noise"]
N_AUGMENTATIONS = 2       # per training track
N_WORKERS       = multiprocessing.cpu_count()  # parallel workers (processes)

# ── Preprocessing ─────────────────────────────────────────────────────────────
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
N_CV_FOLDS      = 5
PCA_VARIANCE    = 0.95    # retain this fraction of variance
K_BEST          = 80      # SelectKBest top-k

# Ensure shared output directories exist
for _d in [PLOTS_EDA, CACHE_DIR, RUNS_ROOT]:
    os.makedirs(_d, exist_ok=True)


def create_run_dir(tag=None):
    """Create a timestamped run directory under ``outputs/runs/``.

    Returns a dict of per-run paths that override the module-level defaults.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts}_{tag}" if tag else ts
    run_dir = os.path.join(RUNS_ROOT, name)

    paths = {
        "RUN_DIR":        run_dir,
        "RESULTS_DIR":    os.path.join(run_dir, "results"),
        "MODELS_DIR":     os.path.join(run_dir, "models"),
        "PLOTS_MODELS":   os.path.join(run_dir, "plots", "models"),
        "SPLITS_DIR":     os.path.join(run_dir, "splits"),
        "CV_RESULTS_CSV": os.path.join(run_dir, "results", "cv_results.csv"),
        "RESULTS_CSV":    os.path.join(run_dir, "results", "final_results.csv"),
        "TRAIN_SPLIT":    os.path.join(run_dir, "splits", "train.csv"),
        "TEST_SPLIT":     os.path.join(run_dir, "splits", "test.csv"),
        "LOG_FILE":       os.path.join(run_dir, "run.log"),
    }

    for _d in [paths["RESULTS_DIR"], paths["MODELS_DIR"],
               paths["PLOTS_MODELS"], paths["SPLITS_DIR"]]:
        os.makedirs(_d, exist_ok=True)

    return paths
