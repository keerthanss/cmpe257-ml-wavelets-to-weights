"""
main.py — end-to-end pipeline for the GTZAN genre classification project.

    python main.py [--force] [--skip-eda] [--models gnb,svm,xgb]

Stages (sequential):
  1. EDA               — exploratory plots           (eda.py)
  2. Feature extraction — original track features     (features.py)
  3. Train/test split   — 80/20 stratified            (cross_validation.py)
  4. Augmentation       — augmented training features  (features.py)
  5. Cross-validation   — per-fold CV + grid search    (cross_validation.py)
  6. Final evaluation   — test-set eval + model saving (models/evaluate.py)

Each run produces a timestamped directory under outputs/runs/ so that
previous results are never overwritten.
"""

import argparse
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from config import RANDOM_STATE, TEST_SIZE, create_run_dir
from features import collect_files, extract_all_features, extract_augmented_features
from cross_validation import load_and_split, grid_search_cv
from models import gnb, knn, svm, rf, xgb, set_plots_dir
from models.evaluate import final_evaluation
from utils import get_logger, init_run_logging, section, fmt_elapsed

log = get_logger("main")

ALL_MODELS = {
    "gnb": gnb,
    "knn": knn,
    "svm": svm,
    "rf":  rf,
    "xgb": xgb,
}


def run_pipeline(force=False, skip_eda=False, selected_models=None):
    pipeline_t0 = time.perf_counter()

    # ── Resolve which models to run ───────────────────────────────────────────
    if selected_models is None:
        models_to_run = list(ALL_MODELS.values())
    else:
        models_to_run = []
        for key in selected_models:
            key_lower = key.strip().lower()
            if key_lower not in ALL_MODELS:
                log.warning(f"Unknown model key '{key}' — skipping "
                            f"(valid: {', '.join(ALL_MODELS)})")
                continue
            models_to_run.append(ALL_MODELS[key_lower])
        if not models_to_run:
            log.error("No valid models selected. Exiting.")
            return None

    # ── Create per-run output directory ───────────────────────────────────────
    if selected_models is not None:
        model_tag = "+".join(k.strip().lower() for k in selected_models
                            if k.strip().lower() in ALL_MODELS)
    else:
        model_tag = "all"
    run_paths = create_run_dir(tag=model_tag)
    init_run_logging(run_paths["LOG_FILE"])
    set_plots_dir(run_paths["PLOTS_MODELS"])

    section(log, "GTZAN Genre Classification — Full Pipeline")
    log.info(f"Run directory: {run_paths['RUN_DIR']}")
    log.info(f"Models selected: {[m.MODEL_NAME for m in models_to_run]}")

    # ── Stage 1: EDA ──────────────────────────────────────────────────────────
    if skip_eda:
        log.info("Skipping EDA (--skip-eda)")
    else:
        section(log, "Stage 1 — Exploratory Data Analysis")
        import eda
        eda.main()

    # ── Stage 2: Feature extraction ───────────────────────────────────────────
    section(log, "Stage 2 — Feature Extraction (original tracks)")
    all_files = collect_files()
    log.info(f"Valid tracks: {len(all_files)}")
    df = extract_all_features(all_files, force=force)

    # ── Stage 3: Train/test split ─────────────────────────────────────────────
    section(log, "Stage 3 — Train/Test Split")
    le_tmp = LabelEncoder()
    y_tmp = le_tmp.fit_transform(df["label"].values)
    train_idx, _ = train_test_split(
        np.arange(len(y_tmp)), test_size=TEST_SIZE,
        stratify=y_tmp, random_state=RANDOM_STATE,
    )
    df_train_for_aug = df.iloc[train_idx].reset_index(drop=True)
    log.info(f"Training tracks (for augmentation): {len(df_train_for_aug)}")

    # ── Stage 4: Augmentation ─────────────────────────────────────────────────
    section(log, "Stage 4 — Augmented Feature Extraction")
    extract_augmented_features(df_train_for_aug, force=force)

    # ── Stage 5: Cross-validation + grid search ──────────────────────────────
    section(log, "Stage 5 — Cross-Validation & Grid Search")
    df_train, df_test, df_aug, feat_cols, le = load_and_split(
        run_paths=run_paths,
    )

    model_configs = [
        (mod.MODEL_NAME, fs, aug, mod.build_model, mod.PARAM_GRID)
        for mod in models_to_run
        for fs in mod.FEAT_SPACES
        for aug in [True, False]
    ]
    log.info(f"Total experiment configs: {len(model_configs)} "
             f"({len(models_to_run)} models × "
             f"{len(models_to_run[0].FEAT_SPACES)} spaces × 2 aug modes)")

    cv_results  = []
    all_cv_rows = []

    for model_name, feat_space, use_aug, build_fn, param_grid in model_configs:
        aug_label = "aug" if use_aug else "no-aug"
        section(log, f"{model_name} [{feat_space}, {aug_label}]")

        try:
            best_params, cv_mean, cv_std, fold_scores = grid_search_cv(
                build_fn, param_grid,
                df_train, df_aug, feat_cols, le,
                feat_space=feat_space,
                use_aug=use_aug,
            )
        except Exception as exc:
            log.warning(f"Skipping {model_name} [{feat_space}, {aug_label}]: "
                        f"{exc}")
            continue

        cv_results.append({
            "model_name":  model_name,
            "feat_space":  feat_space,
            "use_aug":     use_aug,
            "best_params": best_params,
            "build_model": build_fn,
            "cv_mean":     cv_mean,
            "cv_std":      cv_std,
        })

        row = {
            "model":      model_name,
            "feat_space": feat_space,
            "augmented":  use_aug,
            "params":     str(best_params),
            "cv_mean":    cv_mean,
            "cv_std":     cv_std,
        }
        for i, s in enumerate(fold_scores, 1):
            row[f"fold_{i}"] = s
        all_cv_rows.append(row)

    cv_csv = run_paths["CV_RESULTS_CSV"]
    cv_df = pd.DataFrame(all_cv_rows)
    cv_df.to_csv(cv_csv, index=False)
    log.info(f"CV results saved → {cv_csv}")

    # ── Stage 6: Final evaluation ─────────────────────────────────────────────
    section(log, "Stage 6 — Final Test Evaluation")
    summary = final_evaluation(
        df_train, df_test, df_aug, feat_cols, le, cv_results,
        run_paths=run_paths,
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - pipeline_t0
    log.info("=" * 60)
    log.info(f"  PIPELINE COMPLETE")
    log.info(f"  Total elapsed: {fmt_elapsed(elapsed)}")
    log.info(f"  Run directory: {run_paths['RUN_DIR']}")
    log.info("=" * 60)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GTZAN Genre Classification — Full Pipeline",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-extract features even if caches exist")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA plots (useful on re-runs)")
    parser.add_argument(
        "--models", type=str, default=None,
        help=("Comma-separated list of models to run "
              f"(choices: {', '.join(ALL_MODELS)}). "
              "Omit to run all."),
    )
    args = parser.parse_args()
    model_keys = args.models.split(",") if args.models else None
    run_pipeline(force=args.force, skip_eda=args.skip_eda,
                 selected_models=model_keys)
