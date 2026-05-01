"""Random Forest classifier definition."""

from sklearn.ensemble import RandomForestClassifier
from config import RANDOM_STATE

MODEL_NAME   = "Random Forest"
FEAT_SPACES  = ["raw", "pca", "lda", "skb"]
PARAM_GRID  = {
    "n_estimators":     [100, 300, 500],
    "max_features":     ["sqrt", "log2"],
    "min_samples_split": [2, 5],
}


def build_model(**params):
    return RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, **params,
    )
