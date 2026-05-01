"""XGBoost classifier definition."""

from config import RANDOM_STATE

MODEL_NAME   = "XGBoost"
FEAT_SPACES  = ["raw", "pca", "lda", "skb"]
PARAM_GRID  = {
    "learning_rate": [0.1],
    "n_estimators":  [300],
    "subsample":     [0.8],
    "max_depth":     [4],
}


def build_model(**params):
    from xgboost import XGBClassifier
    return XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
