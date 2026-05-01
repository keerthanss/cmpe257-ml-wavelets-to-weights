"""Gaussian Naive Bayes classifier definition."""

from sklearn.naive_bayes import GaussianNB

MODEL_NAME   = "Gaussian NB"
FEAT_SPACES  = ["raw", "pca", "lda", "skb"]
PARAM_GRID   = {}


def build_model(**params):
    return GaussianNB(**params)
