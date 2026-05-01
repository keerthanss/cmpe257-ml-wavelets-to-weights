"""
cross_validation.py — manual k-fold cross-validation with per-fold scaling,
dimensionality reduction, and augmentation stacking.

No sklearn cross_val_score or GridSearchCV.  No feature extraction.
Operates entirely on cached DataFrames loaded once at startup.
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score

from config import (
    FEAT_CACHE, AUG_CACHE, RANDOM_STATE, TEST_SIZE,
    N_CV_FOLDS, TRAIN_SPLIT, TEST_SPLIT,
)
from reduction import fit_scaler, get_reducer, apply_reducer
from utils import get_logger, timed, section

log = get_logger("cross_validation")


# ── Data loading & splitting ──────────────────────────────────────────────────

def load_and_split(run_paths=None):
    """Load feature caches and perform the 80/20 stratified split.

    Parameters
    ----------
    run_paths : dict or None
        Per-run path overrides (from ``create_run_dir()``).

    Returns
    -------
    df_train, df_test, df_aug, feat_cols, le
    """
    train_split = run_paths["TRAIN_SPLIT"] if run_paths else TRAIN_SPLIT
    test_split  = run_paths["TEST_SPLIT"]  if run_paths else TEST_SPLIT

    section(log, "Loading cached features and splitting data")

    df = pd.read_csv(FEAT_CACHE)
    df_aug = pd.read_csv(AUG_CACHE)
    log.info(f"Loaded features: {df.shape[0]} original, "
             f"{df_aug.shape[0]} augmented samples")

    feat_cols = [c for c in df.columns if c not in ("label", "filename")]
    log.info(f"Feature columns: {len(feat_cols)}")

    le = LabelEncoder()
    y_all = le.fit_transform(df["label"].values)

    idx = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE,
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    df_train.to_csv(train_split, index=False)
    df_test.to_csv(test_split, index=False)
    log.info(f"Train split saved → {train_split}  ({len(df_train)} samples)")
    log.info(f"Test split saved  → {test_split}   ({len(df_test)} samples)")
    log.info(f"Classes: {list(le.classes_)}")

    return df_train, df_test, df_aug, feat_cols, le


# ── Augmentation lookup ───────────────────────────────────────────────────────

def _match_augmented(fold_filenames, df_aug, feat_cols):
    """Find augmented rows whose source track is in *fold_filenames*.

    Augmented filenames follow the pattern ``{orig}_aug{idx}_{type}``,
    so splitting on ``_aug`` recovers the original filename.
    """
    train_set = set(fold_filenames)
    aug_base = df_aug["filename"].str.split("_aug").str[0]
    mask = aug_base.isin(train_set)
    matched = df_aug[mask]
    return matched[feat_cols].values, matched["label"].values


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(model_factory, df_train, df_aug, feat_cols, le,
                   feat_space="raw", use_aug=True, n_folds=N_CV_FOLDS):
    """Manual k-fold CV with per-fold scaling, reduction, and optional aug stacking.

    Parameters
    ----------
    model_factory : callable
        Returns an unfitted sklearn-compatible estimator.
    df_train : DataFrame
        Training split (features + label + filename).
    df_aug : DataFrame
        All pre-computed augmented features (ignored when *use_aug* is False).
    feat_cols : list[str]
        Numeric feature column names.
    le : LabelEncoder
        Fitted encoder for string labels -> ints.
    feat_space : str
        One of ``"raw"``, ``"pca"``, ``"lda"``, ``"skb"``.
    use_aug : bool
        Whether to stack augmented data onto each fold's training set.
    n_folds : int
        Number of stratified folds.

    Returns
    -------
    np.ndarray of per-fold accuracy scores.
    """
    X = df_train[feat_cols].values
    y = le.transform(df_train["label"].values)
    filenames = df_train["filename"].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=RANDOM_STATE)
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        t0 = time.perf_counter()

        X_fold_tr,  X_fold_val  = X[tr_idx], X[val_idx]
        y_fold_tr,  y_fold_val  = y[tr_idx], y[val_idx]
        fold_fnames = filenames[tr_idx]

        # 1. Scale on fold-train only
        scaler = fit_scaler(X_fold_tr)
        X_fold_tr_s  = scaler.transform(X_fold_tr)
        X_fold_val_s = scaler.transform(X_fold_val)

        # 2. Reduce on fold-train only
        reducer = get_reducer(feat_space, X_fold_tr_s, y_fold_tr)
        X_fold_tr_r  = apply_reducer(reducer, X_fold_tr_s)
        X_fold_val_r = apply_reducer(reducer, X_fold_val_s)

        # 3. Optionally match and stack augmented rows
        n_aug = 0
        if use_aug:
            X_aug_raw, y_aug_labels = _match_augmented(
                fold_fnames, df_aug, feat_cols,
            )
            if len(X_aug_raw) > 0:
                y_aug = le.transform(y_aug_labels)
                X_aug_s = scaler.transform(X_aug_raw)
                X_aug_r = apply_reducer(reducer, X_aug_s)
                X_train_final = np.vstack([X_fold_tr_r, X_aug_r])
                y_train_final = np.concatenate([y_fold_tr, y_aug])
                n_aug = len(X_aug_raw)
            else:
                X_train_final = X_fold_tr_r
                y_train_final = y_fold_tr
        else:
            X_train_final = X_fold_tr_r
            y_train_final = y_fold_tr

        # 4. Train and score
        model = model_factory()
        model.fit(X_train_final, y_train_final)
        y_pred = model.predict(X_fold_val_r)
        acc = accuracy_score(y_fold_val, y_pred)
        fold_scores.append(acc)

        elapsed = time.perf_counter() - t0
        log.info(f"  Fold {fold_idx}/{n_folds}: "
                 f"train={len(y_train_final)} "
                 f"(orig={len(y_fold_tr)}, aug={n_aug}), "
                 f"val={len(y_fold_val)}, "
                 f"acc={acc*100:.2f}%  [{elapsed:.1f}s]")

    scores = np.array(fold_scores)
    log.info(f"  CV result: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
    return scores


# ── Grid search ───────────────────────────────────────────────────────────────

@timed(log)
def grid_search_cv(build_fn, param_grid, df_train, df_aug, feat_cols, le,
                   feat_space="raw", use_aug=True, n_folds=N_CV_FOLDS):
    """Manual grid search over *param_grid* using :func:`cross_validate`.

    Parameters
    ----------
    build_fn : callable(**params) -> estimator
        Model factory that accepts keyword hyperparameters.
    param_grid : dict
        Hyperparameter grid (empty dict means no tuning).
    use_aug : bool
        Whether to stack augmented data during CV.

    Returns
    -------
    best_params : dict
    best_mean   : float
    best_std    : float
    best_scores : np.ndarray (per-fold)
    """
    if not param_grid:
        log.info("No hyperparameters to tune — running single CV pass")
        scores = cross_validate(
            build_fn, df_train, df_aug, feat_cols, le,
            feat_space=feat_space, use_aug=use_aug, n_folds=n_folds,
        )
        return {}, scores.mean(), scores.std(), scores

    grid = list(ParameterGrid(param_grid))
    log.info(f"Grid search: {len(grid)} parameter combinations × "
             f"{n_folds} folds")

    best_params, best_mean, best_std, best_scores = None, -1.0, 0.0, None

    for i, params in enumerate(grid, 1):
        log.info(f"  Combo {i}/{len(grid)}: {params}")

        def factory(_p=params):
            return build_fn(**_p)

        scores = cross_validate(
            factory, df_train, df_aug, feat_cols, le,
            feat_space=feat_space, use_aug=use_aug, n_folds=n_folds,
        )
        mean_score = scores.mean()

        if mean_score > best_mean:
            best_params = params
            best_mean   = mean_score
            best_std    = scores.std()
            best_scores = scores

    log.info(f"  Best params: {best_params}")
    log.info(f"  Best CV: {best_mean*100:.2f}% ± {best_std*100:.2f}%")
    return best_params, best_mean, best_std, best_scores
