"""
evaluate.py — final model evaluation on the held-out test set.

For every (model, feature-space, use_aug) configuration that went through CV:
  1. Fit scaler + reducer on the *full* training split.
  2. Optionally scale + reduce augmented data with the same transformers.
  3. Stack train (+ aug if use_aug), retrain with best CV hyperparameters.
  4. Evaluate on the test split (used for the first and only time).
  5. Save the trained model and fitted transformers via joblib.
  6. Generate confusion matrices and a CV-vs-test comparison chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score

from config import MODELS_DIR as DEFAULT_MODELS_DIR, RESULTS_CSV as DEFAULT_RESULTS_CSV
from reduction import fit_scaler, get_reducer, apply_reducer
from models import savefig, confusion_matrix_plot
from utils import get_logger, timed, section

log = get_logger("evaluate")

CMAP_FOR_MODEL = {
    "Gaussian NB":   "Blues",
    "KNN":           "Greens",
    "SVM (RBF)":     "Oranges",
    "Random Forest": "Purples",
    "XGBoost":       "Blues",
}


def _model_slug(model_name):
    return (model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", ""))


@timed(log)
def final_evaluation(df_train, df_test, df_aug, feat_cols, le, cv_results,
                     run_paths=None):
    """Train each model on full train (+aug if flagged) and evaluate on test.

    Parameters
    ----------
    df_train, df_test : DataFrame
        Training and test splits (features + label + filename).
    df_aug : DataFrame
        All pre-computed augmented features.
    feat_cols : list[str]
    le : LabelEncoder
    cv_results : list[dict]
        Each entry has keys: model_name, feat_space, use_aug, best_params,
        build_model (callable), cv_mean, cv_std.
    run_paths : dict or None
        Per-run path overrides.

    Returns
    -------
    DataFrame with columns: model, feat_space, augmented, cv_mean, cv_std, test_acc.
    """
    models_dir  = run_paths["MODELS_DIR"]  if run_paths else DEFAULT_MODELS_DIR
    results_csv = run_paths["RESULTS_CSV"] if run_paths else DEFAULT_RESULTS_CSV
    os.makedirs(models_dir, exist_ok=True)

    section(log, "FINAL TEST SET EVALUATION")
    log.info("Test set used for the first and only time.")

    classes = le.classes_
    X_train_raw = df_train[feat_cols].values
    y_train     = le.transform(df_train["label"].values)
    X_test_raw  = df_test[feat_cols].values
    y_test      = le.transform(df_test["label"].values)

    # Pre-match ALL augmented data for training tracks (used when use_aug=True)
    train_filenames = set(df_train["filename"].values)
    aug_base = df_aug["filename"].str.split("_aug").str[0]
    aug_mask = aug_base.isin(train_filenames)
    X_aug_raw = df_aug[aug_mask][feat_cols].values
    y_aug     = le.transform(df_aug[aug_mask]["label"].values)

    log.info(f"Train samples: {len(y_train)}")
    log.info(f"Aug samples:   {len(y_aug)}")
    log.info(f"Test samples:  {len(y_test)}")

    test_results = []

    for entry in cv_results:
        model_name  = entry["model_name"]
        feat_space  = entry["feat_space"]
        use_aug     = entry["use_aug"]
        best_params = entry["best_params"]
        build_fn    = entry["build_model"]
        cv_mean     = entry["cv_mean"]
        cv_std      = entry["cv_std"]
        aug_label   = "aug" if use_aug else "no-aug"

        section(log, f"{model_name} [{feat_space}, {aug_label}]")
        log.info(f"Best params: {best_params}")

        # 1. Scale on full training data
        scaler = fit_scaler(X_train_raw)
        X_train_s = scaler.transform(X_train_raw)
        X_test_s  = scaler.transform(X_test_raw)

        # 2. Reduce
        reducer = get_reducer(feat_space, X_train_s, y_train)
        X_train_r = apply_reducer(reducer, X_train_s)
        X_test_r  = apply_reducer(reducer, X_test_s)

        # 3. Optionally stack train + aug
        if use_aug and len(y_aug) > 0:
            X_aug_s = scaler.transform(X_aug_raw)
            X_aug_r = apply_reducer(reducer, X_aug_s)
            X_final = np.vstack([X_train_r, X_aug_r])
            y_final = np.concatenate([y_train, y_aug])
            log.info(f"Final training set: {len(y_final)} "
                     f"(orig={len(y_train)}, aug={len(y_aug)})")
        else:
            X_final = X_train_r
            y_final = y_train
            log.info(f"Final training set: {len(y_final)} (original only)")

        # 4. Train
        model = build_fn(**best_params)
        model.fit(X_final, y_final)

        # 5. Evaluate
        y_pred = model.predict(X_test_r)
        acc = accuracy_score(y_test, y_pred)
        log.info(f"Test accuracy: {acc*100:.2f}%")

        # 6. Persist model + transformers
        slug = _model_slug(model_name)
        tag  = f"{slug}_{feat_space}_{aug_label}"

        model_path = os.path.join(models_dir, f"{tag}_final.joblib")
        joblib.dump(model, model_path)
        log.info(f"Model saved   → {model_path}")

        scaler_path = os.path.join(models_dir, f"{tag}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        log.info(f"Scaler saved  → {scaler_path}")

        if reducer is not None:
            reducer_path = os.path.join(models_dir, f"{tag}_reducer.joblib")
            joblib.dump(reducer, reducer_path)
            log.info(f"Reducer saved → {reducer_path}")

        # 7. Confusion matrix
        cmap = CMAP_FOR_MODEL.get(model_name, "Blues")
        confusion_matrix_plot(
            y_test, y_pred, classes,
            f"{model_name} [{feat_space}, {aug_label}] — Test: {acc*100:.2f}%",
            f"final_{tag}.png",
            cmap=cmap,
        )

        test_results.append({
            "model":      model_name,
            "feat_space": feat_space,
            "augmented":  use_aug,
            "cv_mean":    cv_mean,
            "cv_std":     cv_std,
            "test_acc":   acc,
        })

    # ── Save & log summary ────────────────────────────────────────────────────
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(results_csv, index=False)
    log.info(f"\nFinal results saved → {results_csv}")

    log.info("\nFinal Results Summary:")
    log.info("\n" + results_df.sort_values("test_acc", ascending=False)
             .to_string(index=False))

    best = results_df.loc[results_df["test_acc"].idxmax()]
    aug_lbl = "aug" if best["augmented"] else "no-aug"
    log.info("=" * 60)
    log.info(f"  BEST MODEL    : {best['model']} [{best['feat_space']}, {aug_lbl}]")
    log.info(f"  TEST ACCURACY : {best['test_acc']*100:.2f}%")
    log.info("=" * 60)

    _plot_comparison(results_df)
    return results_df


# ── Comparison chart ──────────────────────────────────────────────────────────

def _plot_comparison(results_df):
    """Grouped bar chart: CV accuracy vs test accuracy."""
    df = results_df.sort_values("cv_mean", ascending=False)
    x = np.arange(len(df))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(11, len(df) * 0.7), 5))
    ax.bar(x - w / 2, df["cv_mean"].values * 100, w,
           label="CV Accuracy",
           color=sns.color_palette("muted")[0], alpha=0.85)
    ax.bar(x + w / 2, df["test_acc"].values * 100, w,
           label="Test Accuracy",
           color=sns.color_palette("muted")[1], alpha=0.85)
    ax.set_xticks(x)

    labels = []
    for _, r in df.iterrows():
        aug_lbl = "aug" if r["augmented"] else "no-aug"
        labels.append(f"{r['model']}\n[{r['feat_space']}, {aug_lbl}]")
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CV vs Test Accuracy — All Models")
    ax.legend()
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i - w / 2, row["cv_mean"] * 100 + 0.5,
                f"{row['cv_mean']*100:.0f}", ha="center", fontsize=6)
        ax.text(i + w / 2, row["test_acc"] * 100 + 0.5,
                f"{row['test_acc']*100:.0f}", ha="center", fontsize=6)
    plt.tight_layout()
    savefig("final_cv_vs_test.png")
