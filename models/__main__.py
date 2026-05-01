"""
Entry point for model training & evaluation (standalone).

    cd scripts && python -m models

Expects features.csv and augmented_features.csv to already exist in
outputs/cache/.  For the full pipeline (including feature extraction and
EDA), use ``python main.py`` instead.

Each run produces a timestamped directory under outputs/runs/.
"""

import time
import pandas as pd
import numpy as np

from config import create_run_dir
from cross_validation import load_and_split, grid_search_cv
from models import gnb, knn, svm, rf, xgb, set_plots_dir
from models.evaluate import final_evaluation
from utils import get_logger, init_run_logging, section, fmt_elapsed

log = get_logger("models.__main__")

ALL_MODELS = [gnb, knn, svm, rf, xgb]


def _build_model_configs():
    """Return a list of (model_name, feat_space, use_aug, build_fn, param_grid).

    Every model is tested on every feature space x {aug, no-aug}.
    """
    return [
        (mod.MODEL_NAME, fs, aug, mod.build_model, mod.PARAM_GRID)
        for mod in ALL_MODELS
        for fs in mod.FEAT_SPACES
        for aug in [True, False]
    ]


def main():
    pipeline_t0 = time.perf_counter()

    run_paths = create_run_dir()
    init_run_logging(run_paths["LOG_FILE"])
    set_plots_dir(run_paths["PLOTS_MODELS"])

    section(log, "Model Training & Evaluation Pipeline")
    log.info(f"Run directory: {run_paths['RUN_DIR']}")

    df_train, df_test, df_aug, feat_cols, le = load_and_split(
        run_paths=run_paths,
    )

    model_configs = _build_model_configs()
    log.info(f"Total experiment configs: {len(model_configs)}")

    cv_results   = []
    all_cv_rows  = []

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
            "feat_space":  feat_space,
            "augmented":   use_aug,
            "params":      str(best_params),
            "cv_mean":     cv_mean,
            "cv_std":      cv_std,
        }
        for i, s in enumerate(fold_scores, 1):
            row[f"fold_{i}"] = s
        all_cv_rows.append(row)

    # ── Save CV results ───────────────────────────────────────────────────────
    cv_csv = run_paths["CV_RESULTS_CSV"]
    cv_df = pd.DataFrame(all_cv_rows)
    cv_df.to_csv(cv_csv, index=False)
    log.info(f"CV results saved → {cv_csv}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    summary = final_evaluation(
        df_train, df_test, df_aug, feat_cols, le, cv_results,
        run_paths=run_paths,
    )

    elapsed = time.perf_counter() - pipeline_t0
    log.info("=" * 60)
    log.info(f"  PIPELINE COMPLETE")
    log.info(f"  Total elapsed: {fmt_elapsed(elapsed)}")
    log.info(f"  Run directory: {run_paths['RUN_DIR']}")
    log.info("=" * 60)
    return summary


if __name__ == "__main__":
    main()
