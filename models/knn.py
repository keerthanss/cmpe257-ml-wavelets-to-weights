"""K-Nearest Neighbors classifier definition.

KNN is evaluated across multiple feature spaces to find the best combination
of (k, feature_space).
"""

from sklearn.neighbors import KNeighborsClassifier

MODEL_NAME   = "KNN"
FEAT_SPACES  = ["raw", "pca", "lda", "skb"]
PARAM_GRID   = {"n_neighbors": list(range(1, 21))}


def build_model(**params):
    return KNeighborsClassifier(**params)
