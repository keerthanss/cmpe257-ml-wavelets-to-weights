"""Support Vector Machine (RBF kernel) classifier definition."""

from sklearn.svm import SVC
from config import RANDOM_STATE

MODEL_NAME   = "SVM (RBF)"
FEAT_SPACES  = ["raw", "pca", "lda", "skb"]
PARAM_GRID  = {
    "C":     [0.1, 1, 10, 100],
    "gamma": ["scale", "auto"],
}


def build_model(**params):
    return SVC(kernel="rbf", class_weight="balanced",
               random_state=RANDOM_STATE, **params)
