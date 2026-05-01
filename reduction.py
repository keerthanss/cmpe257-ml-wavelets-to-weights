"""
reduction.py — dimensionality reduction utilities.

Provides composable fit/transform functions for StandardScaler, PCA, LDA,
and SelectKBest.  Used per-fold inside cross_validation.py and once for the
final evaluation in models/evaluate.py.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif

from config import PCA_VARIANCE, K_BEST, RANDOM_STATE
from utils import get_logger

log = get_logger("reduction")


# ── Individual fitters ────────────────────────────────────────────────────────

def fit_scaler(X_tr):
    """Fit a StandardScaler on the training split and return it."""
    scaler = StandardScaler()
    scaler.fit(X_tr)
    log.debug(f"Scaler fitted on {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
    return scaler


def fit_pca(X_tr, variance=PCA_VARIANCE):
    """Fit PCA retaining *variance* fraction of explained variance."""
    pca = PCA(n_components=variance, svd_solver="full", random_state=RANDOM_STATE)
    pca.fit(X_tr)
    log.debug(f"PCA: {X_tr.shape[1]} → {pca.n_components_} components "
              f"({variance*100:.0f}% var)")
    return pca


def fit_lda(X_tr, y_tr):
    """Fit LDA for supervised dimensionality reduction."""
    lda = LDA()
    lda.fit(X_tr, y_tr)
    n_out = lda.transform(X_tr[:1]).shape[1]
    log.debug(f"LDA: {X_tr.shape[1]} → {n_out} discriminant components")
    return lda


def fit_skb(X_tr, y_tr, k=K_BEST):
    """Fit SelectKBest (ANOVA F-test) keeping *k* features."""
    skb = SelectKBest(score_func=f_classif, k=k)
    skb.fit(X_tr, y_tr)
    log.debug(f"SelectKBest: {X_tr.shape[1]} → {k} features")
    return skb


# ── Convenience helpers ───────────────────────────────────────────────────────

def get_reducer(feat_space, X_tr_scaled, y_tr):
    """Fit and return the reducer for a named feature space.

    Returns None for "raw" (no reduction needed).
    """
    if feat_space == "raw":
        return None
    elif feat_space == "pca":
        return fit_pca(X_tr_scaled)
    elif feat_space == "lda":
        return fit_lda(X_tr_scaled, y_tr)
    elif feat_space == "skb":
        return fit_skb(X_tr_scaled, y_tr)
    else:
        raise ValueError(f"Unknown feature space: {feat_space}")


def apply_reducer(reducer, X):
    """Transform *X* using a fitted reducer.  Pass-through when reducer is None."""
    if reducer is None:
        return X
    return reducer.transform(X)
